# 2D-homo-lumo 项目全览

> 分子多轮优化 GRPO 训练。基座 Qwen3-8B + LoRA rank=32，数据集 ~5万 prompts。
> 目标：优化分子的 HOMO 和 LUMO 能级。从 2D-qed-logp-mw 的 QED/LogP/MW 任务迁移而来。
> 状态：Reward 设计已完成，待 HOMO/LUMO 计算模型就位后开始训练。

---

## 1. 与 QED/LogP/MW 任务的关键差异

| 维度 | QED/LogP/MW | HOMO-LUMO |
|------|------------|-----------|
| 属性数 | 3 (QED, LogP, MW) | **2** (HOMO, LUMO) |
| 属性单位 | 无/无量纲/Da | **eV** (能量单位) |
| 归一化 | QED有[0,1]边界 | **无天然边界，分母=1** |
| NTE 典型值 | 1.0-1.5 | **2.6** (约2倍) |
| 数据集 | ~163k | **~50k** |
| 方向模式 | 通常单向 | 67%同向, 33%反向 |
| Q_cap | 2.0 | **5.0** |
| Q-gating | 严格改善 | **软门控 (0.05 eV tolerance)** |

---

## 2. 数据分布

### HOMO-LUMO delta 统计 (50,420 prompts)

| 统计量 | HOMO abs Δ | LUMO abs Δ | 合计 NTE |
|--------|-----------|-----------|---------|
| mean | 1.62 | 1.01 | 2.62 |
| std | 1.35 | 0.83 | 1.78 |
| median (P50) | 1.25 | 0.81 | 2.24 |
| P75 | 2.35 | 1.46 | 3.59 |
| P90 | 3.62 | 2.16 | 5.11 |
| P95 | 4.36 | 2.64 | 6.10 |
| max | 9.18 | 6.41 | 13.06 |

### 方向分布

| 模式 | 占比 |
|------|------|
| HOMO↓ LUMO↓ | 34.4% |
| HOMO↑ LUMO↑ | 32.6% |
| HOMO↓ LUMO↑ | 16.5% |
| HOMO↑ LUMO↓ | 16.4% |

---

## 3. 算法核心 (HOMO-LUMO 版)

### 3.1 NTE

```
NTE = |ΔHOMO| + |ΔLUMO|

归一化: 分母=1 (HOMO和LUMO天然同单位eV, 直接相加)
典型值: 初始 mean=2.62, P90=5.11
收敛目标: ~0.15 (学长3-hop结果: HOMO MAE=0.06, LUMO MAE=0.095)
```

### 3.2 Reward 函数

```
r_turn = r_format + r_property + r_rep

r_format:   严格匹配格式 → +1.0, 否则 -10.0
r_property: 无效分子 → -10.0
            有效分子 → Δ_weight × Δ_clip + (Q_weight × Q if soft_gate else 0)
r_rep:      -0.1 × (同prompt内重复数 - 1)
```

**Δ 和 Q 的定义**:
```
Δ_rel = (old_nte - new_nte) / max(old_nte, 1e-6)
Δ_clip = clip(Δ_rel, -4.0, 1.0)
Q = 1.0 - clip(new_nte / 5.0, 0, 1.0)    ← Q_cap=5.0 (P90水平)
```

### 3.3 软 Q-gating (HOMO版独有)

```
QED版:    q_term = Q_weight × Q if improved else 0.0   (严格改善才给Q)
HOMO版:   q_term = Q_weight × Q if new_nte < old_nte + 0.05 else 0.0

0.05 eV tolerance 的意义:
  → 高NTE (5.0): 0.05=1%   → 严格惩罚退化
  → 低NTE (0.15): 0.05=33% → 允许小幅波动，防止噪声被惩罚
  → 天然自适应
```

### 3.4 Reward 数值验证

| 场景 | old NTE | new NTE | improved? | Q-gated? | r_prop |
|------|---------|---------|-----------|----------|--------|
| 大幅改善 | 4.0 | 1.5 | ✅ | ✅ | **1.33** |
| 大幅改善→达标 | 4.0 | 0.2 | ✅ | ✅ | **1.91** |
| 精调改善 | 0.2 | 0.1 | ✅ | ✅ | **1.48** |
| 超精调 | 0.16 | 0.10 | ✅ | ✅ | **1.36** |
| 躺平(同分子) | 0.3 | 0.3 | ❌ | ❌ | **0.00** |
| 微小退化 | 0.15 | 0.18 | ❌ | ✅ (0.03<0.05) | **0.76** |
| 明显退化 | 0.30 | 0.50 | ❌ | ❌ | **-0.67** |
| 大幅恶化 | 2.0 | 6.0 | ❌ | ❌ | **-2.00** |
| 无效分子 | — | — | ❌ | ❌ | **-10.00** |

### 3.5 Accept/Reject + Rollback

```
NTE(new) < NTE(old)?
  → YES: accept, 新分子成为下轮起点
  → NO:  reject, rollback到旧分子

Q-gating 不影响 accept/reject 判定 (仍然严格改善)。
软门控只影响 Q 项是否计入 reward。
```

### 3.6 Tolerance (环境 feedback 用)

```python
TOLERANCE = {"HOMO": 0.06, "LUMO": 0.10}
# 对应 NTE ≈ 0.16, 对齐学长 3-hop MAE
```

### 3.7 NTE 分桶 (仅 metrics)

```python
low:   NTE ≤ 1.0   (P10, ~10%数据)
mid:   1.0 < NTE ≤ 4.0  (P10-P75, ~65%数据)
high:  NTE > 4.0   (P75+, ~25%数据)
```

---

## 4. 共享基础设施 (从 QED/LogP/MW 继承)

以下设计经验证的优秀，HOMO-LUMO 版本直接沿用：

| 组件 | 说明 |
|------|------|
| Turn级GRPO | 按turn_index分组归一化，解决信用分配 |
| PPO+ForwardKL | clip=[0.8,1.2], β分段 0.03→0.05→0.08 |
| LR Schedule | warmup(0→1000) 5e-6→1e-5, hold(1000→6000), cosine→5e-7 |
| Q-gating 思路 | 从QED版继承，HOMO版改软门控 |
| 早停 | 移除，max_turns=3 |
| Logging | clip%, ρ_mean, turn groups |

---

## 5. 超参速查

| 参数 | QED版值 | HOMO版值 | 说明 |
|------|--------|---------|------|
| NTE 公式 | `|ΔQ|/0.5+|ΔL|/3.0+|ΔM|/50.0` | `|ΔHOMO|+|ΔLUMO|` | 待实现 |
| Q_cap | 2.0 | **5.0** | P90水平 |
| Q-gating | `improved` (严格) | `new<old+0.05` (软) | 待实现 |
| Tolerance | QED=0.05,LogP=0.20,MW=5.0 | HOMO=0.06,LUMO=0.10 | 待实现 |
| low/mid/high | ≤0.5/≤2.0/>2.0 | ≤1.0/≤4.0/>4.0 | 待实现 |
| Δ weight | 1.0 | 1.0 | 同 |
| Q weight | 1.0 | 1.0 | 同 |
| Δ_clip | [-4,1] | [-4,1] | 同 |
| Format penalty | -10.0 | -10.0 | 同 |
| Invalid penalty | -10.0 | -10.0 | 同 |
| LR | 5e-6→1e-5→5e-7 | 同 | 待验证 |
| β schedule | 0.03→0.05→0.08 | 同 | 待验证 |
| Turn级GRPO | ✅ | ✅ | 同 |
| 早停 | ❌ | ❌ | 同 |
| K (traj/prompt) | 8 | 8 | 同 |
| T (max turns) | 3 | 3 | 同 |
| γ | 1.1 | 1.1 | 同 |

---

## 6. 待办

- [ ] 等待 HOMO/LUMO 计算模型（从学长获取）
- [ ] 在 `rl_core.py` 中实现：`_property_values()`, `property_error()`, `_parse_property_targets()`, `_target_progress_lines()`
- [ ] 更新 `TARGET_TOLERANCE`, `ABS_NTE_CAP`, `_INITIAL_NTE_BUCKETS`
- [ ] 实现软 Q-gating (`new_nte < old_nte + 0.05`)
- [ ] 更新 `dataset.py` → `DEFAULT_DATASET_NAME`
- [ ] 运行单元测试验证

---

## 7. 相关文档

- `2D-qed-logp-mw/README.md` — QED/LogP/MW 任务文档（完整训练经验参考）
- `training_metrics_reference.md` — 逐行指标详解（共享）
- `training_troubleshooting_guide.md` — 故障排查（共享）
- `~/.claude/plans/jiggly-inventing-lagoon.md` — 改动plan记录
