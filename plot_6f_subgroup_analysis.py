import matplotlib.pyplot as plt
import numpy as np


import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))


# Data for the plot
means = np.array([
    [-0.2, 0.13],  # No adjustment (as decimal)
    [-0.15, 0.10],  # Adjustment with benchmark
    [-0.06, 0.12]   # Adjustment with FAF DL model
])
errors = np.array([
    [0.05, 0.04],  # Confidence intervals for No adjustment
    [0.03, 0.02],  # Confidence intervals for Adjustment with benchmark
    [0.02, 0.03]   # Confidence intervals for Adjustment with FAF DL model
])

# Labels for adjustments
adjustments = [
    "No adjustment", 
    "Adjustment with benchmark", 
    "Adjustment with FAF DL model"
]

# Y positions for the adjustments
y_positions = np.arange(len(adjustments))[::-1]

# Create the figure and axes
fig, ax = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

# Titles for subplots
titles = ["High Dose", "Low Dose"]

# Plot for each subplot
for col in range(2):
    for i, (mean, error) in enumerate(zip(means[:, col], errors[:, col])):
        ax[col].errorbar(mean, y_positions[i], xerr=error, fmt='o', color='black', capsize=0, markersize=10, linewidth=1.8)
        ax[col].text(mean, y_positions[i] + 0.2, f"{int(np.abs(mean) * 100)}%", va='center', ha='center', fontsize=25)



    # Add vertical zero line
    ax[col].axvline(0, color='black', linestyle='--', linewidth=1)

    # Set axis limits and labels
    ax[col].set_xlim(-0.5, 0.51)
    ax[col].set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax[col].set_xticklabels([f"{int(x*100)}%" for x in [-0.5, -0.25, 0, 0.25, 0.5]], fontsize=20)
    ax[col].set_ylim(-0.1, np.max(y_positions) + 0.5)
    # ax[col].set_title(titles[col], fontsize=12)
    # ax[col].set_xlabel("Treatment delta (80% CI)", fontsize=12)

# Shared y-axis for adjustments
ax[0].set_yticks(y_positions)
ax[0].set_yticklabels(adjustments, fontsize=12)
ax[1].set_yticks(y_positions)

ax[1].set_yticklabels([])  # No labels on the right plot
# remove upper and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

# Add directional arrows using ax.annotate()
ax[0].annotate("Favoring treatment", xy=(-0.25, len(adjustments)), xytext=(-0.5, len(adjustments)),
               fontsize=12, ha="center", va="bottom", arrowprops=dict(facecolor="black", arrowstyle="->"))
ax[1].annotate("Favoring Sham", xy=(0.25, len(adjustments)), xytext=(0.5, len(adjustments)),
               fontsize=12, ha="center", va="bottom", arrowprops=dict(facecolor="black", arrowstyle="->"))

# remove upper
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

# remove y-axis spines
ax[1].spines['left'].set_visible(False)
# remove y-axis ticks
ax[1].tick_params(axis='y', which='both', left=False)
# set x-axis label for the whole figure
# fig.text(0.5, 0.01, 'Treatment delta (80% CI)', ha='center', fontsize=20)
# Adjust layout
plt.tight_layout()
plt.subplots_adjust(top=0.98)  # Leave space for the arrows
plt.show()

# save figure
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6f.pdf'), dpi=300)
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6f.png'), dpi=300)