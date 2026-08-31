# 2D-qed-logp-mw: 分子多轮优化 GRPO 训练

> Qwen3-8B + LoRA rank=32 | ~163k prompts | GRPO + PPO + Forward KL
>
> **核心目标**: 通过多轮 fragment 编辑，让 LLM 学会将分子属性 (QED, LogP, MW) 优化到目标值。
>
> **当前状态**: 0-1000 轮训练完成，泛化检验通过（100 targets × 10 trials, 68% NTE 改善）

---

## 1. 文件地图

| 文件 | 职责 |
|------|------|
| `train_grpo_delta_abs.py` | 主入口：参数解析、组装训练管线 |
| `TURN2Bonus0.25KL_delta_abs_rl_core.py` | 核心引擎：全部 RL 逻辑 (~2400 行) |
| `dataset.py` | 数据加载：HF 下载/本地缓存 |
| `model_utils.py` | Tinker 客户端、Tokenizer 管理 |
| `metrics_io.py` | JSONL/CSV/JSON 序列化 |
| `plotting.py` | 训练监控图生成 |
| `prepare_dataset.py` | 一次性数据预下载 |
| `prepare_tokenizer.py` | 一次性 Tokenizer 预下载 |
| `generalization_test.py` | 泛化评估脚本 |
| `tests/test_training_logic.py` | 7 个单元测试 |
| `training_logic_and_reviewer_analysis.md` | 方法完整描述 + 审稿人视角分析 |

---

## 2. 数据流

```
Alan123/qa_dataset (HF, ~163k)
  → _sample_prompt_indices(): 随机采 6→保留 4 个可解析 prompt
    → run_mode_b(): 每个 prompt K=8 条并行轨迹, 最多 T=3 轮编辑
      → 每轮: 采样 → compute_reward → accept/reject → rollback
    → GRPO 组内优势归一化 → datum 构造
  → 汇总 4 prompts × ≤8 traj × ≤3 turns
  → PPO + Forward KL loss → gradient update
  → 每 4 轮 sync 权重到 sampling client
```

---

## 3. 算法核心

### 3.1 NTE (Normalized Total Error)

```
NTE = |ΔQED|/0.5 + |ΔLogP|/3.0 + |ΔMW|/50.0
```

### 3.2 Reward 函数

```
r_turn = r_fmt + r_prop + r_rep + r_early_stop

r_fmt:    +1.0 (格式正确) / -10.0 (格式错误)
r_prop:   -5.0 (无效分子) / 2.0×Δ_clipped + 0.5×Q (有效分子, 统一公式)
r_rep:    -0.1 × (同 prompt 组内重复 canonical SMILES 数 - 1)
r_early:  +2.0 (NTE ≤ 0.25×initial_nte 时触发, 提前终止)
```

**Δ 和 Q 的定义**:
```
Δ_rel = (old_nte - new_nte) / max(old_nte, 1e-6)    # 相对改善
Δ_clip = clip(Δ_rel, -4.0, 1.0)                     # 防极端值
Q = 1.0 - clip(new_nte / 2.0, 0, 1.0)              # 绝对质量 ∈ [0,1]
```

**关键设计**: 统一公式对所有有效分子（无论是否改善）计算相同的 reward。Reward 的符号和大小反映了改善/恶化的程度——连续梯度信号。Rollback 机制独立保证安全（不恶化）。

### 3.3 Accept/Reject + Rollback

```
NTE(new) < NTE(old)?
  → YES: accept, 新分子成为下轮起点
  → NO:  reject, rollback 到旧分子, 注入 rollback feedback
```

Rollback 保证每条轨迹单调不恶化。终端分子 = 轨迹中 NTE 最小的分子。

### 3.4 GRPO 优势

```
每组 K=8 条轨迹: G_k = Σ γ^t × r_turn  (γ=1.1, 后期权重更大)
组内标准化: A_k = (G_k - μ_G) / σ_G
若 σ_G < 1e-8: 全零优势 → 整组丢弃
```

### 3.5 PPO + Forward KL Loss

```
ρ = exp(log π_θ - log π_old)
L_PPO = -min(ρ×A, clip(ρ, 0.8, 1.2)×A)
x = log(π_θ / π_ref)
KL ≈ ρ × (exp(-x) + x - 1)
L_total = L_PPO + β × KL    (β=0.03)
```

### 3.6 LR Schedule

```
warmup 0→1000:  5e-6 → 5e-5
hold 1000→6000: 5e-5
cosine decay 6000→25000: 5e-5 → 1e-6
```

---

## 4. 超参速查

| 参数 | 值 | 
|------|-----|
| LoRA rank | 32 | 
| K (trajectories/prompt) | 8 | 
| T (max turns) | 3 |
| γ (discount) | 1.1 (后期权重更大) |
| β (KL coef) | 0.03 |
| PPO clip | [0.8, 1.2] | 
| Grad clip norm | 1.0 |
| EMA α | 0.05 |
| Temperature | 1.0 |
| Top-p / Top-k | 0.9 / 50 |
| LR warmup/hold/total iters | 1000/5000/25000 |
| LR warmup start | 5e-6 |
| LR max | 5e-5 |
| LR min | 1e-6 |
| Refresh sampling | 每 4 轮 |
| Format penalty | -10.0 |
| Invalid penalty | -5.0 |
| Δ weight | 2.0 |
| Q weight | 0.5 |
| Δ clip range | [-4.0, 1.0] |
| Q cap | 2.0 |
| Early-stop threshold | 0.25 × initial_nte |
| Early-stop bonus | +2.0 |
| Repetition step | -0.1 |

---

## 5. 训练进度 (0-1000 轮)

| Block | mean_nte | accepted | valid | mean_G | forward_kl | high_rel |
|-------|----------|----------|-------|--------|------------|----------|
| 0-99 | 1.077 | 38.5% | 84.6% | -2.13 | 0.005 | 56.4% |
| 100-199 | 1.002 | 42.2% | 92.3% | +0.32 | 0.013 | 62.4% |
| 200-299 | 0.951 | 43.6% | 94.4% | +1.14 | 0.022 | 62.2% |
| 300-399 | 0.948 | 45.9% | 95.4% | +1.76 | 0.032 | 61.7% |
| 400-499 | 0.875 | 48.9% | 97.1% | +2.55 | 0.043 | 65.7% |
| 500-599 | 0.824 | 46.7% | 97.1% | +2.02 | 0.047 | 66.4% |
| 600-699 | 0.779 | 47.6% | 97.3% | +2.29 | 0.052 | 69.2% |
| 700-799 | 0.763 | 49.6% | 97.2% | +2.69 | 0.054 | 71.3% |
| 800-899 | 0.776 | 50.8% | 97.1% | +2.69 | 0.063 | 68.4% |
| 900-999 | 0.764 | 48.6% | 96.9% | +2.63 | 0.071 | 69.2% |

**趋势**: mean_nte 从 1.08 降到 0.76, accepted 率从 38% 升到 49%, validity 从 85% 升到 97%。

---

## 6. 泛化检验结果

### 6.1 评估协议

- **100 个目标分子** × **10 次独立试验** × 最多 **3 轮编辑**
- 报告 **best-of-10** (每条轨迹的最低 NTE)
- 使用 RUN2 checkpoint (global=1000, 训练 1000 轮)
- 与训练 pipeline 完全一致：相同的 prompt 格式、accept/reject/rollback、采样参数

### 6.2 总体结果

| 指标 | 值 |
|------|-----|
| 评估目标数 | 100 |
| 总试验数 | 1000 |
| 初始 NTE (mean) | 3.652 |
| **Best NTE (mean)** | **1.014** |
| Best NTE (median) | 0.925 |
| **相对改善 (mean)** | **67.9%** |
| 相对改善 (median) | 72.3% |
| 改善目标 (>1%) | **99/100 (99.0%)** |
| 未改善 (±1%) | 1/100 (1.0%) |
| 变差 (<-1%) | **0/100 (0.0%)** |

### 6.3 改善幅度分布

```
[ 0%,  25%):   1  ▏
[25%,  50%):  17  ████████████████
[50%,  75%):  39  ██████████████████████████████████████
[75%,  90%):  33  ████████████████████████████████
[90%, 100%):  10  ██████████
```

### 6.4 按初始难度分层

| 难度 | N | 初始 NTE | Best NTE | 相对改善 |
|------|---|---------|----------|---------|
| Mid (0.5-2] | 15 | 1.44 | 0.68 | 52.9% |
| High (>2) | 85 | 4.04 | 1.07 | 70.6% |

### 6.5 各属性误差分析

| 属性 | Mean | Median | P25 | P75 | P90 |
|------|------|--------|-----|-----|-----|
| \|dQED\| | 0.148 | 0.092 | 0.041 | 0.215 | 0.360 |
| \|dLogP\| | 1.556 | 1.245 | 0.572 | 2.307 | 3.494 |
| \|dMW\| | 9.947 | 6.189 | 2.507 | 10.948 | 23.363 |

### 6.6 属性相对改善（中位数）

```
QED:  46.3% 改善
LogP: 47.3% 改善
MW:   93.3% 改善  ← 改善最显著
```

### 6.7 达标率 (within tolerance)

| 属性 | Tolerance | 达标 | 达标率 |
|------|-----------|------|--------|
| QED | ≤0.02 | 14/100 | 14% |
| LogP | ≤0.20 | 8/100 | 8% |
| MW | ≤5.0 | 41/100 | 41% |
| **三项同时** | — | **0/100** | **0%** |

### 6.8 Turn 级表现

| Turn | Valid | Accepted |
|------|-------|----------|
| Turn 1 | 92% | 78% |
| Turn 2 | 94% | 50% |
| Turn 3 | 95% | 36% |

- 总 valid rate: 93.7%
- 总 accept rate: 54.6%
- 平均每 trial: 3.0 turns, 1.6 accepted

### 6.9 NTE 中哪个属性贡献最大？

```
QED:  30.9%   ← 次要贡献
LogP: 47.2%   ← 主导贡献（最难优化）
MW:   21.9%   ← 最小贡献（最容易优化）
```

### 6.10 关键结论

1. **泛化可靠**: 99% 分子得到改善, 0% 被破坏。Rollback 机制保证了分子安全。
2. **MW 最容易优化** (中位数 93% 改善), 因为 fragment 编辑直接改变分子量。
3. **LogP 是瓶颈** (贡献 47% 残留 NTE), 因为 LogP 对局部基团变化极为敏感。
4. **0/100 三项全部达标**——精确同时命中三个属性在化学上极其困难，但 NTE 改善 68% 说明模型在正确方向上大幅逼近目标。
5. **Turn 行为健康**: accept rate 从 78%→50%→36% 递减, 符合边际收益递减规律。

---

## 7. 已解决的算法问题

| 问题 | 解决方案 |
|------|---------|
| 分桶 reward 缺乏理论依据 | 删除 Low/Mid/High 分桶, 统一 reward 公式 |
| Milestone bonus 过于 ad-hoc | 删除 milestone 系统 |
| Flat -3.0 penalty 丢失梯度信息 | 对所有有效分子统一计算公式, rollback 负责安全 |
| GRPO std=0 时 EMA fallback 引入偏差 | 改为全零优势 + 整组丢弃 |
| γ=0.85 前期权重过大 | 改为 γ=1.1, 后期精确编辑权重更大 |
| Early-stop bonus 太小 (1.0) | 提升至 2.0, 约等于一个 turn 满奖励 |
| Format penalty = Invalid penalty | 区分: format=-10, invalid=-5 |
| 无梯度裁剪 | AdamParams(grad_clip_norm=1.0) |
| KL reference 随 checkpoint 漂移 | 固定为 SFT model, 所有训练段共用 |

---

## 8. 关键洞察

### 8.1 Forward KL 解释

| forward_kl | 含义 |
|------------|------|
| 0.000~0.005 | 几乎没学, 策略≈SFT |
| 0.01~0.03 | 学新策略, 偏离可控 |
| 0.03~0.07 | 策略显著偏离, 方向正确 |
| 0.07~0.12 | 偏离较大, 需关注 ρ_mean 和格式退化 |
| >0.15 | 危险, 可能遗忘基础 |

**判断规则**: forward_kl 绝对值不重要, 重要的是 ρ_mean=1.0 且格式/有效性不退化。

### 8.2 Turn 有效性"毕业后"模式

```
0-100轮: 76% valid → 24% 的 turn 在学语法
500+轮:  97% valid → 语法已毕业, RL 专注教策略
```

### 8.3 Rollback 的安全保障

1000 条泛化试验中, 0 条最终分子变差。Rollback 让模型在安全边界内探索。

---

## 9. 分段训练规范

```
段长:   固定 500 轮/段
KL:     固定 0.03
KL ref: 固定为 SFT checkpoint, 永远不变
LR:     从 start_iteration 自动计算
RNG:    --rng-seed <start_iteration>
```

```bash
KL_REF="tinker://da3b0ce4-fbcc-5619-8304-7752622fdc4e:train:2/weights/general-sft-1000"

# 段1 (0-500)
python 2D-qed-logp-mw/train_grpo_delta_abs.py \
  --checkpoint "${KL_REF}" --kl-reference-checkpoint "${KL_REF}" \
  --start-iteration 0 --num-iterations 500 --save-name "rl-500" \
  --kl-penalty-coef 0.03 --rng-seed 0 --allow-overwrite

# 段2 (500-1000) 
python 2D-qed-logp-mw/train_grpo_delta_abs.py \
  --checkpoint <rl-500> --kl-reference-checkpoint "${KL_REF}" \
  --start-iteration 500 --num-iterations 500 --save-name "rl-1000" \
  --kl-penalty-coef 0.03 --rng-seed 500 --allow-overwrite  # KL ref 还是 SFT!
```
