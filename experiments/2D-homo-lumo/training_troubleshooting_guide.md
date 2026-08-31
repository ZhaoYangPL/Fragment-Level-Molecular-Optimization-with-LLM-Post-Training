# GRPO 训练故障排查手册

> 对照你的 KL 日志行和主日志行，逐症状排查。

---

## 症状 1：clip% 突然 > 30%

```
kl | ... | clip%=0.357 | ρ_mean=1.45
```

**根因**：LR 太大，新旧策略差异过大，PPO clip 在强行压制。

**立即操作**：
1. 不要停训练（clip 就是在保护你），但记录当前 iteration
2. 如果同时 ρ_mean > 1.5 → 立刻停，这轮梯度已经在破坏模型
3. 如果 clip%=30~50% 但 ρ_mean < 1.3 → 可以让当前 segment 跑完

**下次调整**：
- LR 降 2-3x：`--lr-max` 从 5e-6 降到 2e-6
- 或保持 LR 不变，增大 KL 系数：`--kl-penalty-coef` 从 0.03 到 0.05

---

## 症状 2：ρ_mean 持续偏离 [0.8, 1.2]

```
ρ_mean 从 1.00 逐步升到 1.35（连续多轮）
```

**根因**：策略在持续单向漂移，KL 约束不够强。

**立即操作**：
- ρ_mean > 1.3 且持续 50+ 轮 → 停训练，从上一个 checkpoint 恢复
- ρ_mean < 0.7 → 策略在坍缩，立刻停

**下次调整**：
- 增大 `--kl-penalty-coef`（0.03 → 0.05 → 0.10）
- 或降低 LR

**判断 β 是否合适**：看 forward_kl
```
forward_kl < 0.001  → β 可能太小（约束太松）
forward_kl 0.001~0.01 → 健康范围
forward_kl > 0.05    → β 太大（约束太紧，学不动）
```

---

## 症状 3：ppo loss 突然 > 1.0 或变成 NaN

```
ppo=1.234 或 ppo=nan
```

**根因**：数值不稳定，通常是某条轨迹的 advantage 极大/极小导致。

**立即操作**：
- NaN → 立刻停，这轮梯度无效且可能破坏模型
- 突然 > 1.0 → 观察下一轮是否恢复，连续 3 轮 > 1.0 就停

**下次调整**：
- 检查是否有 trajectory 拿到了极端 advantage（> 3σ）
- 降低 LR
- 增大 KL β

---

## 症状 4：mean_nte 不降反升（持续 100+ 轮）

```
terminal nte | mean=3.50 → 3.80 → 4.10 ... (上升趋势)
```

**根因**：模型没学到改善分子，或者 reward 信号引导了错误方向。

**诊断步骤**：
1. 看 `turn_valid_rate`：如果 < 50%，模型根本不会生成有效分子 → 回到预训练/SFT 阶段加强
2. 看 `mean_reward_property`：如果持续为负，说明大部分 turn 都在恶化或无效 → Q-gating 可能过度惩罚了探索
3. 看 `pos_adv_ratio`：如果 ≈ 50%，说明 GRPO 没有有效区分好坏 → 组内方差太小

**下次调整**：
- turn_valid_rate 低 → 更多 format 训练数据，或降低 temperature（让采样更保守）
- mean_reward_property 持续为负 → 降低 INVALIDITY_PENALTY（-10 → -5），减小惩罚力度
- pos_adv_ratio ≈ 50% → 增大 K（trajectories_per_prompt 8→16），增加组内方差

---

## 症状 5：pos_adv_ratio 趋近 0% 或 100%

```
主行: +adv%=0.0%  或  +adv%=100.0%
```

**根因**：组内所有轨迹 return 完全相同 → GRPO 失去区分能力。

**0% 的场景**：std_G < 1e-8，触发 EMA fallback，所有 adv 归零
**100% 的场景**：所有 G_k 完全相同且 > EMA baseline

**立即操作**：
- 查看 `dropped_zero_adv` 是否接近 total trajectories
- 如果大量 drop → 这些数据没参与训练，本轮几乎是空转

**下次调整**：
- 增大 K（8→16）：更多轨迹 → 更大组内方差
- 减小 temperature（1.0→0.8）：让采样更多样
- 检查 reward 函数是否缺乏区分度（所有 turn 都拿相同的 reward）

---

## 症状 6：mean_reward_repetition 持续 < -0.3

```
repetition=-0.456
```

**根因**：多条轨迹输出相同的 canonical SMILES → 模式坍缩。

**立即操作**：
- 观察 `final_canonical_smiles` 中是否有大量重复
- 如果 > 50% 轨迹输出同一分子 → 模型已坍缩

**下次调整**：
- 增大 temperature（0.8→1.2）
- 增大 top_p（0.9→0.95）
- 增大 repetition_penalty_step（0.1→0.2）
- 减少 trajectories_per_prompt（16→8）——更多≠更好如果都坍缩

---

## 症状 7：forward_kl 持续上升

```
forward=0.0004 → 0.0023 → 0.0089 → 0.0250 ...
```

**根因**：策略在逐步偏离 reference model。小幅上升是正常的（模型在学习），但加速上升说明约束失效。

**判断**：
```
forward_kl < 0.01  → 正常
forward_kl 0.01~0.05 → 关注，观察是否继续上升
forward_kl > 0.05   → 策略偏离过大，考虑加大 β
```

**下次调整**：
- 加大 `--kl-penalty-coef`：0.03→0.05→0.10

---

## 症状 8：turn_valid_rate 很低（< 40%）

```
validity | turns=12/32 (37.5%)
```

**根因**：模型生成了大量无效 SMILES。

**诊断**：
- 看 rejection_reason 分布：`invalid_molecule` vs `no_improvement`
- 如果大部分是 `invalid_molecule` → 模型不会写合法 SMILES
- 如果大部分是 `no_improvement` → 模型会写 SMILES 但不会改善

**立即操作**：
- `invalid_molecule` 为主 → 这是 SFT 阶段的问题，RL 无法修复。回到 SFT 阶段多训练
- `no_improvement` 为主 → RL 方向对，但改善能力不够。降低 INVALIDITY_PENALTY 让模型敢试

---

## 快速决策表

| 你看到 | 判断 | 改什么 |
|--------|------|--------|
| clip% > 30% | LR 太大 | `--lr-max` ÷ 2 |
| ρ_mean > 1.3 | 策略漂移 | `--kl-penalty-coef` × 2 |
| ppo = NaN | 数值崩溃 | 降 LR + 增 β |
| mean_nte ↑ | 没学到 | 检查 valid_rate, 降 penalty |
| +adv% = 0% | 无区分 | 增大 K 或降 temp |
| repetition < -0.3 | 模式坍缩 | 升 temp, 升 top_p |
| forward_kl > 0.05 | KL 失控 | `--kl-penalty-coef` × 2 |
| valid_rate < 40% | 基础能力不足 | 回 SFT 阶段 |
| mean_G 剧烈震荡 | 不稳定 | 降 LR，等稳定后再升 |
