import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('docs', exist_ok=True)

runs = ['xgboost\nbaseline', 'high\nrecall', 'deep\ntrees', 'fast\nlearner']
auc    = [0.912, 0.912, 0.917, 0.910]
recall = [0.809, 0.949, 0.922, 0.914]
precision = [0.855, 0.683, 0.756, 0.747]
f1     = [0.831, 0.794, 0.831, 0.822]

x = np.arange(len(runs))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')

bars1 = ax.bar(x - 1.5*width, auc,       width, label='AUC-ROC',   color='#58a6ff')
bars2 = ax.bar(x - 0.5*width, recall,    width, label='Recall',    color='#3fb950')
bars3 = ax.bar(x + 0.5*width, precision, width, label='Precision', color='#f78166')
bars4 = ax.bar(x + 1.5*width, f1,        width, label='F1 Score',  color='#d2a8ff')

ax.set_ylim(0.6, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(runs, color='white', fontsize=11)
ax.set_ylabel('Score', color='white', fontsize=12)
ax.set_title('FraudShield — Model Comparison (4 XGBoost Runs)', color='white', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(colors='white')
ax.spines[:].set_color('#30363d')
ax.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.7)
ax.set_axisbelow(True)
ax.legend(facecolor='#161b22', labelcolor='white', fontsize=10)

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.003, f'{h:.3f}', ha='center', va='bottom', color='white', fontsize=7)

plt.tight_layout()
plt.savefig('docs/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Chart 1 saved!")

fig2, ax2 = plt.subplots(figsize=(8, 5))
fig2.patch.set_facecolor('#0d1117')
ax2.set_facecolor('#161b22')

models = ['XGBoost\n(deep-trees)', 'PyTorch\nNeural Net']
auc_vals = [0.917, 0.844]
colors = ['#58a6ff', '#f78166']

bars = ax2.bar(models, auc_vals, color=colors, width=0.4)
ax2.set_ylim(0.7, 1.0)
ax2.set_ylabel('AUC-ROC Score', color='white', fontsize=12)
ax2.set_title('XGBoost vs PyTorch on 590K Transactions', color='white', fontsize=13, fontweight='bold')
ax2.tick_params(colors='white')
ax2.spines[:].set_color('#30363d')
ax2.yaxis.grid(True, color='#30363d', linestyle='--', alpha=0.7)
ax2.set_axisbelow(True)
ax2.set_xticklabels(models, color='white', fontsize=12)

for bar, val in zip(bars, auc_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('docs/xgboost_vs_pytorch.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
print("Chart 2 saved!")
