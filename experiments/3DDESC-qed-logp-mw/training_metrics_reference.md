# GRPO 训练指标速查手册

> 对照你的日志行，逐行解释每个指标的含义、健康范围和异常应对。

---

## 日志结构

每轮迭代打印以下行：

```
主行:     13 | -3.7948 | 7.5620 | 0.8360 | 1.0000 | 56.2% | 88 | 1.052e-06 | 34.1
rewards:  total=-1.270 | format=+0.885 | property=-2.276 [-10.000, 1.744] | repetition=+0.000
turn prop: rel_delta=-0.072 | rel_clip=-0.072 | abs_delta=0.161 | abs_nte=1.978 | abs_q=0.231
bonuses:  early_stop=+0.120 | early_stop_count=5
terminal: mean=1.3846 std=0.9201 valid=32 missing=0 | |dQED|=0.0976 | |dLogP|=1.3691 | |dMW|=36.65
nte rel:  mean=0.4276 std=0.2502
buckets:  low<=0.5: n=0/0 | mid(0.5,2]: n=16/16 init=1.529 term=0.919 rel=0.398 | high>2: n=16/16 init=3.566 term=1.850 rel=0.458
validity: terminal=32/32 (100.0%) | turns=67/88 (76.1%) | accepted=37/88 (42.0%) | valid_rejected=30
kl:       coef=0.0300 | forward=0.000409 | ppo=0.011647 | total=0.011659 | tokens=7187 | clip%=0.002 | ρ_mean=1.000
sampling: candidates=8 kept=4 skipped_unparseable=0
```

---

## 1. 主行

```
13 | -3.7948 | 7.5620 | 0.8360 | 1.0000 | 56.2% | 88 | 1.052e-06 | 34.1
▲            ▲         ▲        ▲        ▲        ▲      ▲      ▲           ▲
Iter       mean_G    std_G   |adv|    std_adv  +adv%  #data   LR        time(s)
```

| 字段 | 含义 | 健康范围 | 异常及应对 |
|------|------|---------|-----------|
| **Iter** | 全局迭代数 | — | — |
| **mean_G** | 8条轨迹平均return | 随训练逐步上升 | 持续下降 → 模型退化，查 mean_nte |
| **std_G** | 8条轨迹return标准差 | > 0.5，有区分度 | < 0.1 持续 → 组内无区分，增大K |
| **\|adv\|** | 平均绝对优势 | 0.3~1.0 | < 0.1 → 信号太弱；> 2.0 → 信号太强可能不稳定 |
| **std_adv** | 优势标准差 | 与\|adv\|量级匹配 | ≈ 0 → 所有turn同一信号 |
| **+adv%** | 正优势占比 | 40~60% | 0%或100% → 组内无区分；长期>80% → 可能reward过于宽松 |
| **#data** | 本轮的training tokens数 | 随K和turn数变化 | 突然降到个位数 → 大量turn被drop |
| **LR** | 当前学习率 | 按schedule走 | — |
| **time(s)** | 本轮耗时 | — | 突然变长 → 可能网络/采样问题 |

---

## 2. rewards 行

```
rewards | total=-1.270 | format=+0.885 | property=-2.276 [-10.000, 1.744] | repetition=+0.000
```

| 字段 | 含义 | 健康范围 | 异常及应对 |
|------|------|---------|-----------|
| **total** | 总reward均值(format+property+rep+early_stop) | 期望逐步上升 | 长期为负 → 大量无效/恶化分子，检查turn_valid_rate |
| **format** | 格式reward均值(+1或-10) | > +0.8 (80%+格式正确) | < +0.5 → 模型不会按格式输出，回SFT阶段 |
| **property** | 属性reward均值 | [-2, +1.5] 均可接受 | 长期 < -5 → 大量无效分子；长期 > 0 → 改善占优 |
| **[min, max]** | property reward范围 | min=-10(有无效)，max~+2 | max < 0 → 改善从来没发生过 |
| **repetition** | 重复惩罚均值 | > -0.2 | < -0.3 → 模式坍缩，多条轨迹相同输出 |

**format 和 property 的分解最有诊断价值**：

```
format≈+1.0, property≈-2  → 模型会写格式但分子不行
format≈-5, property≈-5   → 模型连格式都不会
format≈+1.0, property≈+1  → ✅ 理想状态
```

---

## 3. turn property parts 行

```
turn property parts | rel_delta=-0.072 | rel_clip=-0.072 | abs_delta=0.161 | abs_nte=1.978 | abs_q=0.231
```

| 字段 | 公式 | 含义 |
|------|------|------|
| **rel_delta** | (old_nte-new_nte)/old_nte 的 batch均值 | 相对改善幅度（正=改善，负=恶化） |
| **rel_clip** | clip(rel_delta, -4.0, 1.0) 的均值 | clip后的值，防止极端值主导 |
| **abs_delta** | old_nte-new_nte 的 batch均值 | 绝对改善幅度（受初始NTE影响） |
| **abs_nte** | new_nte 的 batch均值 | 新分子的绝对NTE值 |
| **abs_q** | Q = 1-clip(nte/2,0,1) 的均值 | 分子质量分（0=差，1=好） |

**诊断用法**：

```
rel_delta > 0, abs_q 中等 → ✅ 改善中，质量OK
rel_delta > 0, abs_q → 0  → 改善但分子仍差（高NTE场景，正常）
rel_delta ≤ 0, abs_q → 1  → 躺平（Q-gating后Q归零，没问题）
rel_delta < -2, abs_q → 0  → ⚠️ 大量严重恶化，需关注
```

---

## 4. bonuses 行

```
bonuses | early_stop=+0.120 | early_stop_count=5
```

| 字段 | 含义 | 健康范围 |
|------|------|---------|
| **early_stop** | early-stop bonus均值(+2.0或0) | 0.05~0.5 (少量轨迹触发即可) |
| **early_stop_count** | 本轮触发early-stop的turn数 | 1~5 (4 prompts × 8 traj中少数触发) |

**重要**：early_stop_count 太多 → 模型过于保守，改善一次就停。太少 → 难以达到阈值。这是 γ>1 设计的预期行为——鼓励继续改善。

---

## 5. terminal nte 行 ⭐ 最重要

```
terminal nte | mean=1.3846 std=0.9201 valid=32 missing=0 | |dQED|=0.0976 | |dLogP|=1.3691 | |dMW|=36.65
```

**这是衡量"模型到底有没有让分子变好"的核心指标。**

| 字段 | 含义 | 期望趋势 |
|------|------|---------|
| **mean** | 32个终端分子的平均NTE | ⬇ 持续下降 = 训练有效 |
| **std** | NTE标准差 | 随mean下降而缩小 |
| **valid** | 有效终端分子数 | = trajectories总数 (32) |
| **missing** | 无效终端分子数 | = 0 |
| **\|dQED\|** | 终端分子QED误差均值 | ⬇ 持续下降 |
| **\|dLogP\|** | 终端分子LogP误差均值 | ⬇ 持续下降 |
| **\|dMW\|** | 终端分子MW误差均值 | ⬇ 持续下降 |

**三个属性误差的相对大小告诉你瓶颈在哪**：

```
|dLogP| 很大, |dQED|很小 → LogP优化是瓶颈
|dMW| 很大 → MW难优化（可能需要不同的归一化系数）
```

**当 mean_nte 不再下降时**，看具体是哪个属性误差卡住了——那是模型的优化上限。

---

## 6. nte relative improvement 行

```
nte relative improvement | mean=0.4276 std=0.2502
```

| 字段 | 公式 | 含义 |
|------|------|------|
| **mean** | (initial_nte-terminal_nte)/initial_nte 均值 | 相对改善，0.43=改善了43% |
| **std** | 标准差 | 改善的一致性 |

```
mean > 0.3 → ✅ 改善显著
mean 0.1~0.3 → 中等改善
mean < 0.1 → 改善微弱
mean < 0 → ❌ 终端分子比初始更差（严重问题）
```

---

## 7. initial nte buckets 行

```
initial nte buckets | low: n=0/0 | mid: n=16/16 init=1.529 term=0.919 rel=0.398 | high: n=16/16 init=3.566 term=1.850 rel=0.458
```

按初始NTE将任务分三档，观察不同难度下的表现：

| 桶 | 条件 | n=X/Y | init | term | rel |
|----|------|-------|------|------|-----|
| **low** | init NTE ≤ 0.5 | X=有效数, Y=总数 | 初始NTE均值 | 终端NTE均值 | 相对改善 |
| **mid** | 0.5 < init ≤ 2.0 | 同上 | 同上 | 同上 | 同上 |
| **high** | init > 2.0 | 同上 | 同上 | 同上 | 同上 |

**诊断**：

```
high桶 rel > mid桶 rel → 正常（差分子改善空间大）
mid桶 rel ≈ 0 → ⚠️ 中等难度任务改善不了
low桶 n=0 → 说明初始采样过滤掉了简单任务（正常，NTE太低的分子不需要优化）
```

**关键**：`term` 值。high桶的 term=1.85 说明初始NTE=3.57的分子被改善到了NTE=1.85——改善了近一半。

---

## 8. validity 行 ⭐ 第二重要

```
validity | terminal=32/32 (100.0%) | turns=67/88 (76.1%) | accepted=37/88 (42.0%) | valid_rejected=30
```

| 字段 | 含义 | 健康范围 | 异常及应对 |
|------|------|---------|-----------|
| **terminal** | 终端分子有效性 | **必须 100%**（rollback保证） | < 100% → 代码bug |
| **turns** | turn级分子有效性 | > 70% | < 50% → 模型不会写SMILES，回SFT |
| **accepted** | turn级接受率 | 期望 30~60%，**趋势更重要** | 停滞在 < 20% → 模型太保守或不会改善 |
| **valid_rejected** | 有效但被拒的turn数 | 随accepted上升而下降 | 持续很高 → 模型会写分子但不会改善 |

**核心解读**：

```
turns=67/88 (76.1%)  →  24.9% turn输出无效分子
accepted=37/88 (42.0%)  →  有效turn中有 37/(37+30)=55.2% 被接受
valid_rejected=30  →  30个"有效但没改善"被Q-gating处理

88 turns = 37 accepted + 30 valid_rejected + 21 invalid
```

**趋势是关键**——单看一轮意义不大，看 consecutive 变化：

```
          第N轮        第N+100轮      期望
accepted  17.9%   →   42.0%      ✅ 在变好
turns     75.8%   →   76.1%      →  稳定
```

---

## 9. kl 行

```
kl | coef=0.0300 | forward=0.000409 | ppo=0.011647 | total=0.011659 | tokens=7187 | clip%=0.002 | ρ_mean=1.000
```

| 字段 | 含义 | 健康范围 | 危险信号 |
|------|------|---------|---------|
| **coef** | KL惩罚系数β | 0.01~0.10 | — |
| **forward** | Forward KL divergence | < 0.01 | > 0.05 → 策略偏离过大 |
| **ppo** | PPO clipped loss | 0.01~0.5 | > 1.0 或 NaN → 数值不稳定 |
| **total** | ppo + β×kl | 略大于ppo | — |
| **tokens** | 参与loss的token数 | 数千~数万 | 突然降到几百 → 大量数据被drop |
| **clip%** | 被PPO clip的token比例 | < 15% | > 30% → LR太大 |
| **ρ_mean** | importance ratio均值 | 0.9~1.1 | < 0.7 或 > 1.3 → 策略漂移 |

**最重要的两个**：

```
clip% < 5%   + ρ_mean ≈ 1.0  → ✅ 训练稳定
clip% > 30%  + ρ_mean > 1.3  → ❌ 崩溃前兆，降LR或加β
clip% < 1%   + ρ_mean ≈ 1.0  → 策略更新很保守（训练早期正常）
```

---

## 10. prompt sampling 行

```
prompt sampling | candidates=8 kept=4 skipped_unparseable=0
```

| 字段 | 含义 |
|------|------|
| **candidates** | 每轮随机抽取的候选prompt数 |
| **kept** | 保留的可解析prompt数 |
| **skipped_unparseable** | 因无法解析initial NTE而丢弃的prompt数 |

`skipped > 0` 说明数据集中有 SMILES 无法被 RDKit 解析的脏数据。

---

## 快速诊断流程

```
看到异常 → 定位到行 → 对照上表 → 采取行动

Example:
  "mean_nte 连续50轮不降"
    → 看 validity 行: accepted_rate 是否在涨?
        ├── 在涨 → 正常，NTE下降滞后于accepted改善
        └── 不涨 → 看 turn property parts: rel_delta 正还是负?
                    ├── 正 → 改善存在但太小，需更多训练
                    ├── 负 → 模型在恶化分子
                    └── ≈0 → 躺平，看 repetition 是否坍缩

  "ppo > 1.0"
    → 看 kl 行: clip% 多少?
        ├── > 30% → 降 LR
        └── < 10% → 看是否有单条轨迹 G_k 极端异常

  "accepted_rate < 15% 持续50+轮"
    → 看 turn_valid_rate
        ├── < 50% → 模型连有效SMILES都不会写
        └── > 70% → 模型会写但不会改善，可能是任务太难或探索不够
```

---

## 理想训练曲线

```
轮次     mean_nte    accepted%   clip%    ρ_mean
─────────────────────────────────────────────────
0~100    3.5→2.0    15%→25%    <3%      1.00
100~300  2.0→1.5    25%→35%    <5%      1.00~1.02
300~600  1.5→1.2    35%→45%    <10%     1.00~1.05
600~     1.2→??     45%→??     <15%     1.00~1.10
```

如果偏离这个轨道，回到上面表格逐行排查。
