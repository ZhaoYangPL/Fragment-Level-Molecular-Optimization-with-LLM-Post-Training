import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
base_path = Path(__file__).resolve().parent.parent / "experiments" / "2D-deltaNTE" / "out" / "training_metrics"
checkpoints = {
    1800: "KL0.2-1800multiturn_nonsubprocess_single_stage_kl_iters300_prompts163415_samples8_temp1.0_top_p0.9_top_k50_kl0.2",
    1900: "KL0.2-1900multiturn_nonsubprocess_single_stage_kl_iters100_prompts163415_samples8_temp1.0_top_p0.9_top_k50_kl0.2",
    2000: "KL0.2-2000multiturn_nonsubprocess_single_stage_kl_iters100_prompts163415_samples8_temp1.0_top_p0.9_top_k50_kl0.2",
}

# 读取数据
data = {}
for turn, dir_name in checkpoints.items():
    csv_path = base_path / dir_name / "metrics.csv"
    df = pd.read_csv(csv_path)
    data[turn] = df
    print(f"Loaded {turn} turn: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

# 创建综合对比图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('KL=0.2 Training Metrics Comparison (1800, 1900, 2000 turns)', fontsize=16, fontweight='bold')

# 颜色定义
colors = {1800: '#1f77b4', 1900: '#ff7f0e', 2000: '#2ca02c'}
labels = {1800: '1800 turns', 1900: '1900 turns', 2000: '2000 turns'}

# 1. Training Progress (Total Reward)
ax = axes[0, 0]
for turn in [1800, 1900, 2000]:
    df = data[turn]
    if 'mean_reward_total' in df.columns:
        ax.plot(df.index, df['mean_reward_total'], label=labels[turn], color=colors[turn], linewidth=2, alpha=0.8)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Mean Total Reward', fontsize=11)
ax.set_title('Training Progress (Mean Total Reward)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Valid Molecule Count
ax = axes[0, 1]
for turn in [1800, 1900, 2000]:
    df = data[turn]
    if 'num_final_valid' in df.columns:
        ax.plot(df.index, df['num_final_valid'], label=labels[turn], color=colors[turn], linewidth=2, alpha=0.8)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Valid Molecule Count', fontsize=11)
ax.set_title('Final Valid Molecules per Step', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Property Errors (Average of QED, LogP, MW)
ax = axes[1, 0]
for turn in [1800, 1900, 2000]:
    df = data[turn]
    if 'mean_final_qed_error' in df.columns:
        avg_error = (df['mean_final_qed_error'] + df['mean_final_logp_error'] + df['mean_final_mw_error']) / 3
        ax.plot(df.index, avg_error, label=labels[turn], color=colors[turn], linewidth=2, alpha=0.8)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('Average Property Error', fontsize=11)
ax.set_title('Mean Property Errors (QED, LogP, MW)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. KL Divergence
ax = axes[1, 1]
for turn in [1800, 1900, 2000]:
    df = data[turn]
    if 'kl_policy_reference' in df.columns:
        ax.plot(df.index, df['kl_policy_reference'], label=labels[turn], color=colors[turn], linewidth=2, alpha=0.8)
ax.set_xlabel('Step', fontsize=11)
ax.set_ylabel('KL Divergence', fontsize=11)
ax.set_title('KL Policy-Reference Divergence', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图表
output_dir = Path(__file__).resolve().parent.parent / "experiments" / "2D-deltaNTE" / "output" / "training_plots"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "KL0.2-combined_metrics_1800_1900_2000.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved combined metrics plot to: {output_path}")

# 显示统计信息
print("\n=== Data Statistics ===")
for turn in [1800, 1900, 2000]:
    df = data[turn]
    print(f"\n{turn} turns ({labels[turn]}):")
    print(f"  - Total steps: {len(df)}")
    if 'mean_reward_total' in df.columns:
        print(f"  - Final Mean Total Reward: {df['mean_reward_total'].iloc[-1]:.4f}")
    if 'num_final_valid' in df.columns:
        print(f"  - Final Valid Count: {df['num_final_valid'].iloc[-1]:.0f}")

print("\nPlot generated successfully!")
