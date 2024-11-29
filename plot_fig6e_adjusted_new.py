import matplotlib.pyplot as plt
import numpy as np


import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


# this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))

# Example data (replace with your actual data)
months = np.array([0, 6, 12, 18])
yticks = np.arange(0, 4.1, 1)

# sham_original = []

sham = [0, 1, 2.05, 3.1]
low_dose = [0, 0.95, 2, 3.4]
high_dose = [0, 0.7, 1.6, 2.4]
error = [0, 0.2, 0.4, 0.7]  # Example error bars

high_dose_color = "#0E3E87"
low_dose_color = "#F2B035"
# sham_color a darker gray
sham_color = "#808080"
# Create a figure and axes
fig, ax = plt.subplots(1, 2, figsize=(12, 7), sharey=True)

# Left plot (Unadjusted Analysis)
ax[0].errorbar(months-0.5, sham, yerr=error, label='Sham', marker='o', linestyle='-', capsize=3, color=sham_color)
ax[0].errorbar(months, low_dose, yerr=error, label='Low dose', marker='o', linestyle='-', capsize=3, color=low_dose_color)
ax[0].errorbar(months+0.5, high_dose, yerr=error, label='High dose', marker='o', linestyle='-', capsize=3, color=high_dose_color)
ax[0].set_title("Unadjusted analysis", fontsize=20)
ax[0].set_xlabel("Month", fontsize=20)
ax[0].set_ylabel("Mean GA change from baseline (+/- SE)", fontsize=20)
ax[0].set_xticks(months)
ax[0].set_yticks(yticks)
ax[0].set_ylim(-0.1, 4.01)
# set font size
ax[0].tick_params(axis='both', which='major', labelsize=20)

# ax[0].legend(fontsize=20, loc = 'lower right', frameon=False)
ax[0].annotate('~20%\np=0.16*(<0.2)',
               xy=(18.5, 2.4), xytext=(10, 3),
               arrowprops=dict(facecolor='black', shrink=0.05),
               fontsize=16)
# remove upper and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)


# Right plot (FAF Model Adjusted Analysis)
sham_adjusted = [0, 1, 2.05, 3.1]  # Replace with actual adjusted data
low_dose_adjusted = [0, 0.92, 1.9, 3.28]
high_dose_adjusted = [0, 0.95, 1.95, 2.9]
error = [0, 0.1, 0.2, 0.35]  # Example error bars

ax[1].errorbar(months-0.5, sham_adjusted, yerr=error, label='Sham', marker='o', linestyle='-', capsize=3, color=sham_color)
ax[1].errorbar(months, low_dose_adjusted, yerr=error, label='Low dose', marker='o', linestyle='-', capsize=3, color=low_dose_color)
ax[1].errorbar(months+0.5, high_dose_adjusted, yerr=error, label='High dose', marker='o', linestyle='-', capsize=3, color=high_dose_color)
ax[1].set_title("OCTCubEFM adjusted analysis", fontsize=20)
ax[1].set_xlabel("Month", fontsize=20)
ax[1].set_xticks(months)
# set font size
ax[1].tick_params(axis='both', which='major', labelsize=20)
# remove y-axis spines
ax[1].spines['left'].set_visible(False)
# remove y-axis ticks
ax[1].tick_params(axis='y', which='both', left=False)

ax[1].annotate('~4.5%\np=0.6',
               xy=(18.5, 2.9), xytext=(12, 3),
               arrowprops=dict(facecolor='black', shrink=0.05),
               fontsize=16)
# remove upper
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

# Extract legend handles and labels from ax[0]
handles, labels = ax[0].get_legend_handles_labels()

# Add a single legend on top using handles and labels from ax[0]
fig.legend(handles, labels, loc='upper center', fontsize=18, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.025))

# Final adjustments
# plt.suptitle("Comparison of Unadjusted and Adjusted Analyses", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.94])
# plt.show()
# save figure
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6e.pdf'), dpi=300)
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6e.png'), dpi=300)