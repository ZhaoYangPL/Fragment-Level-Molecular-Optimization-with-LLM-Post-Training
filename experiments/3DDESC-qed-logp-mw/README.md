# 3DDESC-qed-logp-mw 项目全览

> 从 2D-qed-logp-mw 派生。在 prompt 中注入 RDKit 计算的 3D 分子描述符作为辅助信息。
> 目标：探究加入 3D 信息后优化效果是否提升（消融实验）。
> 状态：代码就绪，待训练。

---

## 1. 与 2D-qed-logp-mw 的差异

| 维度 | 2D 版 | 3D 版 |
|------|-------|-------|
| Reward 公式 | 不变 | 不变 |
| NTE | 不变 | 不变 |
| PPO/KL/LR | 不变 | 不变 |
| Turn级GRPO | ✅ | ✅ |
| 早停 | ❌ | ❌ |
| **prompt 内容** | 仅 2D 目标 | **+ 3D 描述符** |

**唯一差异**：每个 turn 的 prompt 末尾注入当前分子的 3D 描述符。其他所有逻辑完全不变。

---

## 2. 3D 描述符设计

### 2.1 选取的 6 个描述符 × 3 组

| 组 | 描述符 | 物理含义 | 与目标的天然关系 |
|----|--------|---------|----------------|
| **size** | volume | 范德华体积 (Å³) | 密度≈0.6-0.8 → MW≈0.6×volume |
| | sasa | 溶剂可及表面积 (Å²) | 尺寸信息，辅助 MW |
| **shape** | npr1 | I1/I3 (1=球, 0=棒) | 药物分子近球形 → QED 偏高 |
| | npr2 | I2/I3 (1=球, 0=盘) | 与 npr1 互补的形状指纹 |
| **surface** | hydrophobic_fraction | 疏水表面比例 | 直接影响 LogP |
| | polar_charge_separation | Gasteiger 电荷空间分离 (D) | 影响极性和偶极矩 |

### 2.2 注入格式（纯数据，不指导）

```
Molecular size:
  - volume=234.5 (van der Waals volume, Å³)
  - sasa=312.8 (solvent-accessible surface area, Å²)
Molecular shape:
  - npr1=0.72 (I1/I3, 1=sphere 0=rod)
  - npr2=0.85 (I2/I3, 1=sphere 0=disc)
Surface chemistry:
  - hydrophobic_fraction=0.34 (proportion of hydrophobic surface)
  - polar_charge_separation=2.15 (charge separation, Debye)
```

关键设计原则：
- ✅ 描述符名称 + 数值 + 物理含义解释
- ✅ 按物理本质分组（size/shape/surface）
- ❌ 不标注 low/moderate/high（不需要分位数）
- ❌ 不给行为建议（不写 "prefer removing bulky fragments"）
- ❌ 不绑定优化目标（不写 "MW: volume=... LogP: hydrophobic=..."）
- ❌ 不修改 system prompt
- ❌ 不添加决策提示

### 2.3 计算方式

RDKit 3D 构象生成 (ETKDGv3) + MMFF/UFF 优化 → 描述符计算。
每次 prompt 构造时实时计算，不需要预缓存。

---

## 3. 文件地图

| 文件 | 职责 |
|------|------|
| `three_d_descriptors.py` 🆕 | 3D 描述符计算、context 格式化 |
| `train_grpo_delta_abs.py` ✏️ | 增加 `--three-d-groups` 和 `--three-d-seed` |
| `TURN2Bonus0.25KL_delta_abs_rl_core.py` ✏️ | prompt 注入 + 参数传递 (~15行) |
| 其余文件 | 与 2D 版完全相同 |

---

## 4. 消融实验方案

| 实验 | 命令 | 注入内容 | 检验假设 |
|------|------|---------|---------|
| Baseline | `--three-d-groups none` | 无 | 基线 |
| +Size | `--three-d-groups size` | volume + sasa | 3D 尺寸信息帮助 MW 优化? |
| +Surface | `--three-d-groups surface` | hydrophobic_fraction + polar_charge_separation | 表面化学帮助 LogP? |
| +Shape | `--three-d-groups shape` | npr1 + npr2 | 形状信息帮助 QED? |
| +All | `--three-d-groups all` | 全部 6 个 | 综合效果 |

每组独立跑，对比 mean_nte、accepted_rate、high桶 rel。

---

## 5. 用法

```bash
# 基线（无 3D）
python train_grpo_delta_abs.py --three-d-groups none ...

# 全部 3 组
python train_grpo_delta_abs.py --three-d-groups all ...

# 模块化消融
python train_grpo_delta_abs.py --three-d-groups size ...
python train_grpo_delta_abs.py --three-d-groups shape ...
python train_grpo_delta_abs.py --three-d-groups surface ...
```

其他超参 (β, LR, K, T 等) 与 2D 版完全相同，从 2D 版 checkpoint 加载。

---

## 6. 相关文档

- `2D-qed-logp-mw/README.md` — 原版 2D 任务文档
- `3D-desc/` — 原始 3D 参考代码（注意：该版有 hand-crafted rules，本版已移除）
- `three_d_descriptors.py` — 描述符计算和格式化
