# 分子优化 GRPO 训练逻辑完整说明 & 审稿人视角分析

> 本文档供 AAAI 投稿参考，包含两部分：(A) 方法完整描述 (B) 审稿人视角的批判性分析

---

## Part A: 训练逻辑完整描述

### A.1 问题设定

给定一个分子（SMILES）和目标属性变化量（ΔQED, ΔLogP, ΔMW），让 LLM 通过多轮 fragment-level 编辑逐步优化分子，使其属性逼近目标值。

**数据格式**:
```
Input:  "Given molecule <SMILES>CCO</SMILES>, ..., make QED <QED>0.094</QED> higher,
         LogP <LogP>2.898</LogP> lower, MW <MW>101.104</MW> lower."
Output: "Replace c1ccc2occ()c2c1 with N to form <SMILES>CC(C)(C)NC(=O)CNc1ccccc1F</SMILES>."
```

- 数据集: `Alan123/qa_dataset_with_qed_logp_mw_grpo`，~163k 条
- 基座模型: Qwen3-8B + LoRA (rank=32)
- 每轮采样: 从 6 个候选中筛选 4 个 prompt，每个跑 K=8 条并行轨迹

---

### A.2 多轮轨迹采集

每轮最多 **T=3** 轮编辑。流程如下：

```
Round 1: 初始分子 → 模型生成编辑 → 计算 reward → 判断 accept/reject
Round 2: accept → 用新分子继续; reject → rollback 到上一轮分子
Round 3: 同上
Early-stop: 若 NTE 足够小，提前终止
```

**Accept/Reject 机制**: 当且仅当新分子的 NTE < 旧分子的 NTE 时，该轮编辑被接受（accept），更新当前分子的 SMILES。否则被拒绝（reject），系统 rollback 到上一轮分子，并告知模型编辑被拒的原因。

**Prompt 更新**: 每轮 accept 后，重算当前分子与目标的差距，更新 prompt 中的 `<QED>`, `<LogP>`, `<MW>` 标签值。同时更新 BRICS fragments。

---

### A.3 Normalized Total Error (NTE)

NTE 是核心评估指标，衡量当前分子与目标属性值的加权偏差：

$$
\text{NTE}(mol, target) = \frac{|qed - qed^*|}{0.5} + \frac{|\log P - \log P^*|}{3.0} + \frac{|MW - MW^*|}{50.0}
$$

其中归一化分母 $0.5, 3.0, 50.0$ 分别对应 QED 的理论范围 [0,1]、LogP 的典型范围 [-3, 6]、MW 的典型范围 [100, 500] 的粗略尺度。

**Initial NTE** 是起始分子与目标的 NTE，用于分桶和 milestone 定义。

---

### A.4 Reward 函数

每个 turn 的 reward 由五个组件构成:

#### A.4.1 Format Reward

$$
r_{\text{fmt}} = \begin{cases} +1.0 & \text{输出严格匹配 Replace/Add/Remove <SMILES> 格式} \\ -10.0 & \text{格式不匹配} \end{cases}
$$

#### A.4.2 Property Reward

**输入**: 旧分子 NTE (`old_nte`), 新分子 NTE (`new_nte`)

**两个派生信号**:

- **相对改善** (Relative Delta NTE):
  $$\Delta_{\text{rel}} = \frac{old\_nte - new\_nte}{\max(old\_nte, 10^{-6})}$$

  正值表示改善，clip 到 $[-4.0, 1.0]$

- **绝对质量** (Absolute NTE Quality):
  $$Q = 1.0 - \text{clip}(new\_nte / 2.0,\ 0,\ 1.0)$$

  $Q \in [0, 1]$，越接近 1 表示分子质量越好

**分桶策略**: 根据初始 NTE 将任务分为三类，每类用不同的 property reward 公式:

| 桶 | 条件 | 优化策略 | Property Reward |
|----|------|---------|-----------------|
| **Low** | initial NTE ≤ 0.5 | 保守微调，避免大幅改动 | $r = 0.5Q + 2.0 \cdot \text{clip}(\Delta_{\text{rel}}, -2.0, 0.5)$ |
| **Mid** | 0.5 < initial NTE ≤ 2.0 | 平衡改善与质量 | $r = 2.0 \cdot \Delta_{\text{rel}}^{\text{clip}} + 0.5Q$ |
| **High** | initial NTE > 2.0 | 激进缩小差距 | $r = 1.0 \cdot \Delta_{\text{rel}}^{\text{clip}} + 1.0Q$ |

**特殊情况**:
- 分子无效 → $r = -10.0$
- 分子有效但 NTE 未改善 → $r = -3.0$

**设计动机**: 对初始偏差大的任务（High 桶），鼓励大跨步改善；对已接近目标的任务（Low 桶），避免 overshoot，更重视绝对质量。

#### A.4.3 Repetition Penalty

在同一个 prompt 的 K=8 条轨迹中，若有 $n$ 条最终生成相同的 canonical SMILES:

$$r_{\text{rep}} = -0.1 \times (n - 1)$$

鼓励轨迹多样性，防止模型坍缩到同一输出。

#### A.4.4 Milestone Bonus

三个 NTE 改善里程碑，跨过时获得一次性 bonus:

| Level | 条件 | Bonus |
|-------|------|-------|
| 1 | $current\_nte \leq 0.75 \times initial\_nte$ | +0.25 |
| 2 | $current\_nte \leq 0.50 \times initial\_nte$ | +0.50 |
| 3 | $current\_nte \leq 0.25 \times initial\_nte$ | +0.75 |

每条轨迹追踪 `best_milestone_level`，只在首次跨越时发放。跨越多级则累计（如从 level 0→2 获得 0.25+0.50=0.75）。

#### A.4.5 Early-Stop Bonus

若当前 NTE 降到阈值以下，轨迹提前终止并获 +1.0 bonus:

$$\text{threshold} = \begin{cases} 0.5 & initial\_nte > 2.0 \\ 0.25 \times initial\_nte & initial\_nte \leq 2.0 \end{cases}$$

#### A.4.6 Total Turn Reward

$$r_{\text{turn}} = r_{\text{fmt}} + r_{\text{prop}} + r_{\text{rep}} + r_{\text{milestone}} + r_{\text{early\_stop}}$$

---

### A.5 轨迹 Return 与 GRPO 优势

**折扣 Return** ($\gamma=0.85$):

$$G_k = \sum_{t=0}^{T-1} \gamma^t \cdot r_{\text{turn}, t}^{(k)}$$

**组内标准化**: 在每个 prompt 的 K=8 条轨迹内:

$$A_k = \frac{G_k - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{K}\sum G_k$$

若 $\sigma_G < 10^{-8}$（所有轨迹 return 几乎相同），退化为 EMA baseline:

$$A_k = G_k - \text{EMA}, \quad \text{EMA}_{new} = 0.95 \cdot \text{EMA}_{old} + 0.05 \cdot G_k$$

**Token 级赋值**: 轨迹优势 $A_k$ 赋给所有 completion token，prompt token 赋 0。$|A_k| < 10^{-12}$ 的轨迹被丢弃。

---

### A.6 PPO + Forward KL 损失

#### PPO Clipped Objective

对每个 action token，计算 importance ratio $\rho = e^{\log\pi_\theta - \log\pi_{\text{old}}}$:

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}\left[\min(\rho A,\ \text{clip}(\rho, 0.8, 1.2) \cdot A)\right]$$

#### Forward KL Divergence

使用固定 reference model $\pi_{\text{ref}}$（初始 SFT checkpoint）计算 forward KL:

$$D_{\text{KL}} = \mathbb{E}\left[\rho \cdot \max(e^{-(\log\pi_\theta - \log\pi_{\text{ref}})} + (\log\pi_\theta - \log\pi_{\text{ref}}) - 1,\ 0)\right]$$

这里使用了 non-negative KL 近似（$\exp(-x) + x - 1 \geq 0$ 当 $x \neq 0$），并用 importance ratio $\rho$ 加权。

#### Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{PPO}} + \beta \cdot D_{\text{KL}}, \quad \beta = 0.03$$

---

### A.7 学习率调度

Warmup + Hold + Cosine Decay 三阶段:

| 阶段 | Iter | LR |
|------|------|-----|
| Linear warmup | 0 → 1,000 | 5×10⁻⁶ → 5×10⁻⁵ |
| Constant hold | 1,000 → 6,000 | 5×10⁻⁵ |
| Cosine decay | 6,000 → 25,000 | 5×10⁻⁵ → 1×10⁻⁶ |

---

### A.8 超参一览

| 参数 | 值 | 说明 |
|------|-----|------|
| LoRA rank | 32 | Qwen3-8B |
| Trajectories per prompt (K) | 8 | GRPO 组内样本数 |
| Max turns (T) | 3 | 多轮编辑上限 |
| $\gamma$ | 0.85 | 折扣因子 |
| $\beta$ (KL coef) | 0.03 | Forward KL 权重 |
| PPO clip | [0.8, 1.2] | Importance ratio 裁剪 |
| EMA $\alpha$ | 0.05 | 优势 fallback |
| Temperature | 1.0 | 采样温度 |
| Top-p / Top-k | 0.9 / 50 | 采样策略 |
| Refresh sampling | 每 4 iter | 权重同步频率 |
| Format penalty | -10.0 | 格式错误惩罚 |
| Invalid penalty | -10.0 | 无效分子惩罚 |
| No-improvement penalty | -3.0 | 未改善惩罚 |

---

## Part B: 审稿人视角的批判性分析

> 以下从 AAAI 审稿人角度，对每个算法设计决策提出可能的质疑和修改建议。

---

### B.1 NTE 归一化分母 (0.5, 3.0, 50.0) 的选择

**审稿人可能的质疑**:
- 为什么是 0.5 / 3.0 / 50.0？这些值是先验的还是 data-driven 的？
- 如果是先验的，意味着 QED 偏离 0.5 的"重要性"等于 LogP 偏离 3.0 等于 MW 偏离 50.0。这个等价关系有化学依据吗？
- 不同属性间的 trade-off 在分子设计中是核心问题，但 NTE 将所有属性线性加和，抹平了这种 trade-off。

**修改建议**:
- 引用 QED 的定义范围 [0, 1] 解释 0.5（理论半范围），LogP 引用 DrugBank 统计分布解释 3.0，MW 引用 Lipinski Rule of 5 解释 50.0
- 做消融实验：改变归一化权重（如 [0.3, 3.0, 40.0] vs [0.7, 5.0, 60.0]），展示结果不敏感
- 或从数据集中统计属性标准差，用 data-driven 的归一化系数

---

### B.2 Reward 分桶策略 (Low/Mid/High)

**审稿人可能的质疑**:
- 分桶边界 0.5 和 2.0 是任意选择的（heuristic），缺少理论或实验支撑
- 三个桶的 reward 公式权重不同：Low 桶用 `0.5Q + 2.0Δ`，Mid 用 `2.0Δ + 0.5Q`，High 用 `1.0Δ + 1.0Q`。这些系数组合（0.5/2.0, 2.0/0.5, 1.0/1.0）的对称性看起来很像是手工调出来的
- 为什么不直接用同一个 reward 公式？分桶是否真的必要？

**修改建议**:
- 补充消融实验：对比 "分桶 reward" vs "统一 reward公式 (`2.0Δ + 0.5Q`)" 的效果
- 展示分桶后不同难度任务的改善分布
- 如果消融显示分桶帮助不大，考虑简化为统一公式

---

### B.3 Milestone Bonus System

**审稿人可能的质疑（这是最容易被攻击的点）**:
- Milestone bonus 是 goal-conditioned RL 或 robotics 中常见的技巧，但在分子优化中使用缺乏先例
- 三个层级的阈值 (0.75, 0.50, 0.25) 和对应的 bonus (0.25, 0.50, 0.75) 存在精确的对称关系（0.25+0.75=1, 0.50+0.50=1），这看起来像是人为设计的
- 这本质上是 reward shaping —— 审稿人会问：不 shaping 的 baseline 效果如何？
- Milestone bonus 会改变最优策略：模型可能更关注"跨越阈值"而非"持续改善"。例如：从 initial_nte=10 改善到 7.5（跨 level 1）获得额外 +0.25，但从 7.6 改善到 7.4 获得同样bonus，尽管实际改善较小

**修改建议**:
- 这是最需要消融实验支持的部分：训练 with/without milestone bonus，展示 bonus 对收敛速度和最终性能的影响
- 讨论 milestone 的设计与 NTE 分桶策略之间的关系——两者都基于初始 NTE
- 如果 milestone 的消融效果不显著，考虑移除以简化方法

---

### B.4 Rollback 机制

**审稿人可能的质疑**:
- Rollback 保证每轮不会恶化分子，这类似于 constrained optimization 中的 feasibility 约束。但审稿人可能问：是否过于保守，限制了探索？
- Rollback 后模型收到"你的编辑被拒了"的 feedback，这引入了额外的 prompt engineering。如果 feedback 措辞改变，结果是否稳健？
- 在 RL 文献中，类似的做法是 reward penalty（而非显式 rollback）。为什么不直接用负 reward？

**修改建议**:
- 论证 rollback 在分子优化场景中的必要性：无效分子占比较高（~40%），rollback 是实用的工程选择
- 对比 rollback vs negative-reward-only 的效果
- 讨论 rollback 对学习效率的影响：是否会导致模型过度保守？

---

### B.5 GRPO 组内优势 + EMA Fallback

**审稿人可能的质疑**:
- GRPO 的组内标准化假设同组轨迹来自同一个 prompt，优势衡量的是"相对于其他轨迹的好坏"。但当 K=8 条轨迹全部无效时（这在分子优化中很常见），std 接近 0，退化为 EMA fallback
- EMA fallback 是全局的，但 GRPO 的优势应该是组内相对的。这种混合破坏了 GRPO 的理论性质
- DeepSeekMath 的原始 GRPO 实现在 std=0 时给出全零优势（即这些样本不参与训练），而非 EMA fallback

**修改建议**:
- 统计训练中 EMA fallback 的触发频率。如果 >20%，说明组内方差常为零，GRPO 的组内假设不适用
- 考虑改用全零优势（drop 无信息样本）而非 EMA fallback
- 或增大 K（如 K=16）来增加组内方差

---

### B.6 Forward KL 中的 Importance Ratio 加权

**审稿人可能的质疑**:
- 标准的 forward KL 是 $\mathbb{E}_{\pi_{\text{ref}}}[\log(\pi_{\text{ref}} / \pi_\theta)]$。代码中使用 $\rho \cdot \text{KL}$ 其中 $\rho = \pi_\theta / \pi_{\text{old}}$（而非 $\pi_\theta / \pi_{\text{ref}}$）
- 这意味着 KL 惩罚被 importance ratio 缩放，而 importance ratio 可能很大或很小，导致 KL 项不稳定
- 为什么不用更标准的 reverse KL 或简单的 $\mathbb{E}[\text{KL}]$？

**代码分析**:
```python
policy_reference_logratio = new_action - reference_action  # log(πθ/πref)
nonnegative_kl = expm1(-policy_reference_logratio) + policy_reference_logratio
forward_kl_terms.append(importance_ratio * clamp(nonnegative_kl, min=0.0))
```

其中 `expm1(-x) + x` 是 $\exp(-x) + x - 1$，这是 $\text{KL}(\pi_{\text{ref}} \| \pi_\theta)$ 在 $x = \log(\pi_\theta / \pi_{\text{ref}})$ 时的单个样本估计（Taylor 展开）。$\rho$ 乘在前面是用 importance sampling 纠正分布偏移。

**修改建议**:
- 引用 TRPO/PPO 文献中的 KL 估计方法，说明当前实现的出处
- 监控训练中 `importance_ratio_mean` 是否保持在 [0.8, 1.2] 内。若偏离很大，说明 KL 估计不准确
- 补充与简单 reverse KL（不乘 $\rho$）的对比实验

---

### B.7 PPO Clip Range 与训练效率

**审稿人可能注意到**:
- clip range [0.8, 1.2] 相比原始 PPO 的 [0.8, 1.2] 在合理范围内
- 但如果 clip fraction（被裁剪的 token 比例）持续 > 30%，说明学习率过大或 clip 范围过紧
- 当前代码有 `ppo_clip_fraction` 指标但缺少讨论

**修改建议**:
- 在论文中报告训练过程中的 clip fraction 和 importance_ratio_mean
- 如果 clip fraction 持续 > 50%，说明当前的 clip range 太紧，需要调宽或降低 LR

---

### B.8 Format Penalty 和 Invalid Penalty 相同 (-10.0)

**审稿人可能的质疑**:
- 格式错误（模型输出不符合指定格式）和化学无效（SMILES 不合法）是两种完全不同的错误，但收到相同的 -10 惩罚
- 模型无法从 reward 信号中区分这两种错误——不知道是需要改善格式还是改善化学合理性
- 如果大部分负样本是格式错误，模型可能学到了错误的关联

**修改建议**:
- 区分两种 penalty：format error 保持 -10，invalid molecule 改为 -5 或更温和的值
- 或者对格式错误用更大的惩罚（-20），因为格式是可以直接学习的
- 展示两种错误的比例随训练的变化

---

### B.9 数据效率 — 每次迭代仅 32 条轨迹

**审稿人可能的质疑**:
- 每轮迭代只用 4 prompts × 8 trajectories = 32 条轨迹做一次梯度更新
- 每次更新后丢弃数据（没有 experience replay 或多 epoch 训练）
- 这远低于典型 PPO 实现的 batch size（数千样本）
- 小 batch 导致梯度方差大，训练不稳定

**修改建议**:
- 要么增大 prompts_per_iter（如 16-32）
- 要么在采集的数据上做 2-4 个 epoch 的 mini-batch 训练
- 讨论为什么小 batch 在这个场景下足够（如果确实足够的话）

---

### B.10 缺少验证集和泛化性评估

**审稿人几乎肯定会问**:
- 所有评估指标都在训练集的 prompts 上计算——如何证明模型不是在记忆训练数据？
- 模型在未见过的分子上的表现如何？
- 生成的分子是否真的更好（不仅仅是 NTE 更低），还是只是学会了特定的编辑模式？

**修改建议（必须补充）**:
- 留出 5-10% 数据作为验证集，报告验证集上的 NTE 改善
- 做化学合理性检查：生成的分子是否满足 drug-likeness（QED > 0.5）、合成可及性（SA score）等
- 展示生成分子的化学多样性

---

### B.11 Dead Code 和代码质量问题（审稿人不太可能直接看到，但如果开源会被发现）

上述问题已在最近的代码清理中修复。建议在论文的 supplementary material 中提供干净的代码仓库链接。

---

## Part C: 实验建议优先级

| 优先级 | 实验 | 目的 |
|--------|------|------|
| 🔴 P0 | 训练 500 轮，确保 mean_nte 持续下降 | 证明方法有效 |
| 🔴 P0 | 留出 10% 验证集，报告 val NTE | 证明泛化性 |
| 🟠 P1 | Milestone bonus 消融 | 证明 reward shaping 必要 |
| 🟠 P1 | 分桶 reward vs 统一 reward 消融 | 证明分桶策略必要 |
| 🟠 P1 | 不同 KL β (0.01, 0.03, 0.05, 0.10) 对比 | 证明 β 选择合理 |
| 🟡 P2 | Rollback vs negative-reward 对比 | 展示 rollback 的必要性 |
| 🟡 P2 | Clip range [0.7,1.3] vs [0.8,1.2] 对比 | 展示超参鲁棒性 |
| 🟢 P3 | 化学合理性分析（SA, QED 分布） | 增强化学可解释性 |

---

## 附录: 关键超参配置速查

| 组件 | 参数 | 值 |
|------|------|-----|
| NTE | QED/LogP/MW 归一化 | 0.5 / 3.0 / 50.0 |
| Reward | Δ_weight / Q_weight | 2.0 / 0.5 |
| Reward | Δ clip | [-4.0, 1.0] |
| Reward | Q cap | 2.0 |
| Reward | 分桶边界 | 0.5, 2.0 |
| Milestone | 阈值 × initial_nte | 0.75, 0.50, 0.25 |
| Milestone | Bonus | +0.25, +0.50, +0.75 |
| Early-stop | High bucket threshold | 0.5 |
| Early-stop | Low bucket ratio × initial | 0.25 |
| Early-stop | Bonus | +1.0 |
| PPO | Clip range | [0.8, 1.2] |
| KL | β | 0.03 |
| Discount | γ | 0.85 |
| EMA | α | 0.05 |
| LR | warmup/max/min | 5e-6 / 5e-5 / 1e-6 |
| LR | warmup/hold/total iters | 1000 / 5000 / 25000 |
| Sampling | K / T | 8 / 3 |
