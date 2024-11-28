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


from sklearn.linear_model import LinearRegression


# Step 1: Simulate baseline covariate (e.g., baseline severity)
np.random.seed(42)
n = 50  # Number of samples per group
covariate_control = np.random.normal(0, 1, n)  # Covariate for control group
covariate_treatment = np.random.normal(0, 1, n)  # Covariate for treatment group (centered on zero)

# Step 2: Simulate observed outcomes (with higher variance in raw data)
true_treatment_effect = 1.5  # True difference in treatment effect
y_control = covariate_control + np.random.normal(0, 0.4, n)  # Control group (larger variance)
y_treatment = true_treatment_effect + covariate_treatment + np.random.normal(0, 0.4, n)  # Treatment group
y_observed = np.concatenate([y_control, y_treatment])  # Combine outcomes
group_labels = np.array(['Control'] * n + ['Treatment'] * n)  # Labels

# Step 3: Covariate adjustment (linear regression to reduce variance)
covariate_all = np.concatenate([covariate_control, covariate_treatment]).reshape(-1, 1)
model = LinearRegression()
model.fit(covariate_all, y_observed)  # Fit model to predict outcome based on covariate
y_predicted = model.predict(covariate_all)

# Step 4: Compute residuals (smaller variance after adjustment)
residuals = y_observed - y_predicted

# Plotting
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(1, 3, width_ratios=[1, 2, 1])  # 1:2:1 ratio for subplots

# Axes Layout
ax1 = fig.add_subplot(grid[0])  # First subplot (25% width)
ax2 = fig.add_subplot(grid[1:2])  # Middle subplot (50% width)
ax3 = fig.add_subplot(grid[2])  # Third subplot (25% width)

# Step 4: Compute residuals (smaller variance after adjustment)
residuals = y_observed - y_predicted

# Split residuals into control and treatment groups
residuals_control = residuals[:n]
residuals_treatment = residuals[n:]

# Add jittered x-axis values
jitter_control = np.random.uniform(-0.1, 0.1, n)  # Jitter for control group
jitter_treatment = np.random.uniform(-0.1, 0.1, n)  # Jitter for treatment group

# Update x-axis values for Panel 1 (Observed Outcome)
x_control_jittered = 0 + jitter_control  # Control group centered around 0
x_treatment_jittered = 1 + jitter_treatment  # Treatment group centered around 1
# Plotting
fig = plt.figure(figsize=(12, 6))  # Adjust figure size
grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1])  # Assign 1:2:1 ratios

# Axes Layout
ax1 = fig.add_subplot(grid[0])  # First subplot (25% width)
ax2 = fig.add_subplot(grid[1:3])  # Middle subplot (50% width)
ax3 = fig.add_subplot(grid[3])  # Third subplot (25% width)

# Panel 1: Observed outcome by treatment (with jittered x-axis values)
ax1.scatter(x_control_jittered, y_control, color='darkblue', label='Control', alpha=0.7)
ax1.scatter(x_treatment_jittered, y_treatment, color='blue', label='Treatment', alpha=0.7)
# draw a horizontal line for each group to show the mean for each group (only around each group)
ax1.plot([np.min(x_control_jittered)-0.1, np.max(x_control_jittered)+0.1], [np.mean(y_control), np.mean(y_control)], color='black', linestyle='-', alpha=1, linewidth=2.5)
ax1.plot([np.min(x_treatment_jittered)-0.1, np.max(x_treatment_jittered)+0.1], [np.mean(y_treatment), np.mean(y_treatment)], color='black', linestyle='-', alpha=1, linewidth=2.5)
print(np.mean(y_control) - np.mean(y_treatment))
# ax1.arrow(0.5, np.mean(y_control), 0.7, np.mean(y_treatment) - np.mean(y_control), head_width=0.3, head_length=0.2, fc='orange', ec='orange')
# ax1.set_title("Observed Outcome by Treatment", fontsize=12)
# ax1.set_ylabel("Y Observed", fontsize=10)
ax1.set_xticks([0, 1])
# ax1.set_xticklabels(["Control", "Treatment"])
ax1.set_xticklabels(["", ""])
ax1.set_ylim(-3.5, 3.5)
ax1.set_yticks([])

# ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3])
# ax1.set_yticklabels(["-3", "-2", "-1", "0", "1", "2", "3"], fontsize=15)
# ax1.set_yticks([-3, -1.5, 0, 1.5, 3])
# ax1.set_yticklabels(["-3", "-1.5", "0", "1.5", "3"], fontsize=15)
ax1.grid(False)

# Panel 2: Observed vs Predicted
ax2.scatter(y_predicted[:n], y_observed[:n], alpha=0.7, color='darkblue')
ax2.scatter(y_predicted[n:], y_observed[n:], alpha=0.7, color='blue')
# Draw thin lines connecting each point to the y=x line
for x, y in zip(y_predicted, y_observed):
    ax2.plot([x, x], [x, y], color='darkblue', linestyle='-', linewidth=0.5)

ax2.plot([min(y_predicted), max(y_predicted)], [min(y_predicted), max(y_predicted)], color='darkblue', linestyle='-', label='Identity Line')
ax2.set_xticks([])
ax2.set_yticks([])
# ax2.set_title("Observed vs Predicted", fontsize=12)
# ax2.set_xlabel("Y Predicted", fontsize=10)
# ax2.set_ylabel("Y Observed", fontsize=10)
# ax2.set_xticks([-3, -1.5, 0, 1.5, 3])
# ax2.set_yticks([-3, -1.5, 0, 1.5, 3])
# ax2.set_xticklabels(["-3", "-1.5", "0", "1.5", "3"], fontsize=15)
# ax2.set_yticklabels(["-3", "-1.5", "0", "1.5", "3"], fontsize=15)
# ax2.set_xticklabels(["-3", "0", "3"], fontsize=15)
# ax2.set_yticklabels(["-3", "0", "3"], fontsize=15)
ax2.grid(False)

# Panel 3: Residuals by treatment (with jittered x-axis values)
ax3.scatter(x_control_jittered, residuals_control, color='darkblue', alpha=0.7)
ax3.scatter(x_treatment_jittered, residuals_treatment, color='blue', alpha=0.7)
# ax3.arrow(0.5, np.mean(residuals_control), 0.7, np.mean(residuals_treatment) - np.mean(residuals_control), head_width=0.3, head_length=0.2, fc='orange', ec='orange')
# ax3.set_title("Residuals by Treatment", fontsize=12)
# ax3.set_ylabel(r"$Y_{\text{observed}} - Y_{\text{predicted}}$", fontsize=10)
# draw a horizontal line for each group to show the mean for each group (only around each group)
ax3.plot([np.min(x_control_jittered)-0.1, np.max(x_control_jittered)+0.1], [np.mean(residuals_control), np.mean(residuals_control)], color='black', linestyle='-', alpha=1, linewidth=2.5)
ax3.plot([np.min(x_treatment_jittered)-0.1, np.max(x_treatment_jittered)+0.1], [np.mean(residuals_treatment), np.mean(residuals_treatment)], color='black', linestyle='-', alpha=1, linewidth=2.5)
print(np.mean(residuals_control) - np.mean(residuals_treatment))
ax3.set_xticks([0, 1])
ax3.set_xticklabels(["", ""])
# ax3.set_xticklabels(["Control", "Treatment"])
ax3.grid(False)
ax3.set_ylim(ax1.get_ylim())  # Match y-axis limits with Panel 1
ax3.set_yticks([])

# Remove upper and right spines for cleaner visualization
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# # Panel 1: Observed outcome by treatment (with jittered x-axis values)
# ax1.scatter(x_control_jittered, y_control, color='darkblue', label='Control', alpha=0.7)
# ax1.scatter(x_treatment_jittered, y_treatment, color='blue', label='Treatment', alpha=0.7)
# ax1.arrow(0.5, np.mean(y_control), 0.7, np.mean(y_treatment) - np.mean(y_control), head_width=0.3, head_length=0.2, fc='orange', ec='orange')
# ax1.set_title("Observed Outcome by Treatment", fontsize=12)
# ax1.set_ylabel("Y Observed", fontsize=10)
# ax1.set_xticks([0, 1])
# ax1.set_xticklabels(["Control", "Treatment"])
# ax1.set_ylim(-4, 4)
# ax1.grid(False)



# # Panel 2: Observed vs Predicted (2:1 aspect ratio)
# ax2.scatter(y_predicted[:n], y_observed[:n], alpha=0.7, color='darkblue')
# ax2.scatter(y_predicted[n:], y_observed[n:], alpha=0.7, color='blue')
# ax2.plot([min(y_predicted), max(y_predicted)], [min(y_predicted), max(y_predicted)], color='black', linestyle='--', label='Identity Line')
# ax2.set_title("Observed vs Predicted", fontsize=12)
# ax2.set_xlabel("Y Predicted", fontsize=10)
# ax2.set_ylabel("Y Observed", fontsize=10)
# ax2.set_aspect(2)  # 2:1 width-to-height ratio
# ax2.grid(False)


# # Panel 3: Residuals by treatment (with jittered x-axis values)
# ax3.scatter(x_control_jittered, residuals_control, color='darkblue', alpha=0.7)
# ax3.scatter(x_treatment_jittered, residuals_treatment, color='blue', alpha=0.7)
# ax3.arrow(0.5, np.mean(residuals_control), 0.7, np.mean(residuals_treatment) - np.mean(residuals_control), head_width=0.3, head_length=0.2, fc='orange', ec='orange')
# ax3.set_title("Residuals by Treatment", fontsize=12)
# ax3.set_ylabel(r"$Y_{\text{observed}} - Y_{\text{predicted}}$", fontsize=10)
# ax3.set_xticks([0, 1])
# ax3.set_xticklabels(["Control", "Treatment"])
# ax3.grid(False)

# ax3.set_ylim(ax1.get_ylim())  # Match y-axis limits with Panel 1
# # remove upper and right spines
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# ax3.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
# save figure
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6d_illustrative_covariate_adjust.png'), dpi=300)
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_6d_illustrative_covariate_adjust.pdf'), dpi=300)