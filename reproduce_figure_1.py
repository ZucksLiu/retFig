import os
import pickle
import collections
import pandas as pd
import numpy as np
import math

from fig_settings import *
from fig_utils import *
from matplotlib.colors import LinearSegmentedColormap
from ext_dataset_setting import *
from inhouse_dataset_setting import *
from icd10_dataset_setting import *

PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    "from_scratch 3D": "Supervised (3D, w/o pre-training)",
    "retfound 2D": "RETFound (2D)",
    "from_scratch 2D": "Supervised (2D, w/o pre-training)",
}

plot_col_idx = ['auroc', 'acc', 'auprc', 'bal_acc']
plot_col = 0
plot_name = 'auroc'
plot_col = 2
plot_name = 'auprc'

save_file_dir = os.path.dirname(os.path.abspath(__file__))
retclip_exp_res = os.path.join(save_file_dir, 'retClip_exp_res.csv')
retclip_exp_res_df = pd.read_csv(retclip_exp_res)
print(retclip_exp_res_df)

retclip_exp_res_aireadi = os.path.join(save_file_dir, 'retClip_exp_res_aireadi.csv')
retclip_exp_res_aireadi_df = pd.read_csv(retclip_exp_res_aireadi)
print(retclip_exp_res_aireadi_df)
# exit()

print('inhouse:', len(inhouse_grouped_dict))
print('ext:', len(ext_oph_grouped_dict))
print('icd10:', len(inhouse_non_oph_organized_dict))
print(inhouse_grouped_dict.keys())
res_df_dict = {
                ('MAE-joint', '3D'): collections.OrderedDict(),
                ('retfound', '3D'): collections.OrderedDict(),
                ('retfound', '2D'): collections.OrderedDict(),
                }


for task in inhouse_non_oph_organized_dict['default'].keys():
    for key, value in inhouse_non_oph_organized_dict['default'][task].items():
        value = np.array(value)
        res_df_dict[key][TASKS[task]] = np.mean(value, axis=0)[plot_col]


print(res_df_dict)

for task in inhouse_grouped_dict['fewshot'].keys():
    for key, value in inhouse_grouped_dict['fewshot'][task].items():
        value = value[0]
        print(value)
        if key in res_df_dict:
            res_df_dict[key][task] = value[0][plot_col]

print(res_df_dict)
print(ext_oph_grouped_dict)
for task in ext_oph_grouped_dict['fewshot'].keys():
    for key, value in ext_oph_grouped_dict['fewshot'][task].items():
        value = np.array(value)
        if key in res_df_dict:
            res_df_dict[key][task] = np.mean(value, axis=0)[plot_col]
print(res_df_dict, len(res_df_dict), len(res_df_dict[('MAE-joint', '3D')]))
res_df_dict[('retfound', '2D')]['Maestro2 \n(AI-READI)'] = 0.5208
res_df_dict[('retfound', '3D')]['Maestro2 \n(AI-READI)'] = 0.6382
res_df_dict[('MAE-joint', '3D')]['Maestro2 \n(AI-READI)'] = 0.6627

plot_col_idx = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
plot_col = 0
for idx, row in retclip_exp_res_df.iterrows():
    task = row['Method']
    task_split = task.split(' ')
    if task_split[0] == 'MAE-joint':
        task = (task_split[0], '3D')
    elif task_split[0] == 'retFound':
        task = (task_split[0].lower(), task_split[1])
        

    plot_col_used = 'OCT_to_IR ' + plot_col_idx[plot_col]
    res_df_dict[task][plot_col_used + ' UW-Med'] = row[plot_col_used]
    plot_col_used = 'IR_to_OCT ' + plot_col_idx[plot_col]
    res_df_dict[task][plot_col_used + ' UW-Med'] = row[plot_col_used]
print(res_df_dict) 

for idx, row in retclip_exp_res_aireadi_df.iterrows():
    task = row['Method']
    task_split = task.split(' ')
    if task_split[0] == 'MAE-joint':
        task = (task_split[0], '3D')
    elif task_split[0] == 'retFound':
        task = (task_split[0].lower(), task_split[1])
        

    plot_col_used = 'OCT_to_IR ' + plot_col_idx[plot_col]
    res_df_dict[task][plot_col_used + ' AI-READI'] = row[plot_col_used]
    plot_col_used = 'IR_to_OCT ' + plot_col_idx[plot_col]
    res_df_dict[task][plot_col_used + ' AI-READI'] = row[plot_col_used]

print(res_df_dict, len(res_df_dict[('MAE-joint', '3D')]))

def plot_radar(df, ax, is_fill=True, is_box=False):
    categories = list(df.index)
    N = len(categories) # What will be the angle of each axis in the plot?
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    plt.xticks(angles[:-1], ['' for _ in angles[:-1]])
    #y_min=0.45
    for column in df:
        values = df[column].tolist()

        y_maxs, y_mins = [], []
        for i, angle in enumerate(angles[:-1]):
            cat = categories[i]
            y_max = ceil_to_nearest(np.max(df.loc[cat].to_list()))
            y_min = floor_to_nearest(np.min(df.loc[cat].to_list()), 0.1)
            # round to the nearest 0.05
            #y_max = round(math.ceil(y_max*10) / 10, 1)
            y_maxs.append(y_max)
            y_mins.append(y_min)
        values = [(values[i] - y_mins[i]) / (y_maxs[i] - y_mins[i]) for i in range(len(y_maxs))]
        #values = [values[i] / y_maxs[i] for i in range(len(y_maxs))]
        values += values[:1]
        column_color = column[0] + ' ' + column[1]
        plot_method = PLOT_METHODS_NAME[column_color]
        print('angles:', angles, 'values:', values)
        print(values[0])
        ax.plot(angles, values, linewidth=3, label=plot_method, c=COLORS_RADAR[column_color])
        if is_fill:
            ax.fill(angles, values, alpha=0.1)

    global_max = 1
    n_steps = 5
    #y_min=0.4
    #ax.set_ylim(y_min, global_max)

    for i, angle in enumerate(angles[:-1]):
        cat = categories[i]
        # print(cat)
        # print(categories)
        # exit()
        y_max = ceil_to_nearest(np.max(df.loc[cat].to_list()))
        y_min = y_mins[i]
        y_step = (float(y_max) - y_min) / (n_steps)
        r_ticks = np.arange(3, n_steps + 1) / n_steps
        #ax.set_yticks(r_ticks)
        #ax.set_yticklabels(['' for _ in r_ticks])
        y_labels = ["{:.2f}".format(round((r + 3)*y_step, 2) + y_min) for r in range(len(r_ticks))]
        if angle < np.pi / 2 or angle > 3 * np.pi / 2:
            label_angle = angle * 180 / np.pi
        else:
            label_angle = angle * 180 / np.pi - 180
        for r, l in zip(r_ticks, y_labels):
            ax.text(angle, r, l, rotation=label_angle,rotation_mode='anchor', horizontalalignment='center',verticalalignment='center', fontsize=2*MEDIUM_SIZE)
        if cat.endswith('UW-Med') or cat.endswith('AI-READI'):
            cat = cat.replace('OCT_to_IR', 'OCT2IR')
            cat = cat.replace('IR_to_OCT', 'IR2OCT')
            if cat.endswith('UW-Med'):
                cat = cat.replace(' UW-Med', '\n(UW-Med)')
            if cat.endswith('AI-READI'):
                cat = cat.replace(' AI-READI', '\n(AI-READI)')
        cat = cat.replace('recall@1', '')

        if 'EGFR' in cat or 'FAT1' in cat or 'KRAS' in cat or 'LRP1B' in cat or 'TP53' in cat or 'Exon' in cat or 'L858R' in cat or 'TMB' in cat or 'biomarker' in cat:
            #c = '#4daf4a'
            c = '#998ec3'
            boxstyle='round,pad=0.3,rounding_size=0.4'
        elif 'Sub' in cat:
            #c = '#377eb8'
            c = '#f1a340'
            cat = cat.replace('-Sub', '\nTyping')
            boxstyle='round,pad=0.6,rounding_size=0.4'
        else:
            #c = '#e41a1c'
            c = '#d95f0e'
            cat = cat.replace('-Surv', '')
            boxstyle='round,pad=0.6,rounding_size=0.4'
        if 'TCGA' in cat or 'Exon' in cat:
            dis = r_ticks[-1] + 0.18
            a = angle - 0.001
        elif 'L858R' in cat:
            a = angle - 0.015
        elif 'TMB' in cat:
            a = angle + 0.015
        elif len(cat) <= 5:
            dis = r_ticks[-1] + 0.15
            a = angle - 0.001
        else:
            dis = r_ticks[-1] + 0.2
            a = angle - 0.001
        if cat.startswith('Hypertension'):
            # adjust the position of the text
            dis = r_ticks[-1] + 0.18
            a = angle + 0.04
            print(cat, dis, a)
        if cat == 'IR2OCT \n(AI-READI)':
            print('come in IR2OCT')
            # exit()
            dis = r_ticks[-1] + 0.22
            a = angle + 0.03
        if cat == 'OCT2IR \n(AI-READI)':
            print('come in ')
            # exit()
            dis = r_ticks[-1] + 0.23
            a = angle + 0.035
        if cat == 'IR2OCT \n(UW-Med)':
            # print('come in ')
            # exit()
            dis = r_ticks[-1] + 0.21
            a = angle + 0.055
        if cat == 'OCT2IR \n(UW-Med)':
            # print('come in ')
            # exit()
            dis = r_ticks[-1] + 0.2
            a = angle + 0.06

        if cat == 'Diabetes':
            dis = r_ticks[-1] + 0.22
            # a = angle + 0.05
        if cat.startswith('Hyperlipidemia'):
            dis = r_ticks[-1] + 0.15
            a = angle + 0.0
        if cat.startswith('POG') or cat.startswith('Pain'):
            dis = r_ticks[-1] + 0.12
            a = angle -0.001
        if cat.startswith('Maestro2'):
            dis = r_ticks[-1] + 0.17
            a = angle + 0.02
        if cat.startswith('Soft tissue'):
            dis = r_ticks[-1] + 0.21
            a = angle - 0.07
        if cat.startswith('Back pain'):
            dis = r_ticks[-1] + 0.13
            a = angle - 0.01
        if not is_box:
            ax.text(a, dis, cat, rotation=0, \
                rotation_mode='anchor', horizontalalignment='center',verticalalignment='center', color='k',fontsize=2.5*MEDIUM_SIZE)
        else:
            ax.text(a, dis, cat, rotation=0, \
                rotation_mode='anchor', horizontalalignment='center',verticalalignment='center', color='k',fontsize=6*MEDIUM_SIZE, \
                    bbox=dict(facecolor='none', edgecolor=c,boxstyle=boxstyle, linewidth=2))
            #bbox=dict(facecolor='none', edgecolor=c,boxstyle=boxstyle, linewidth=2))
        print('cat:', cat)
        ax.set_rgrids(r_ticks, ['' for _ in r_ticks], angle=angle*180/np.pi, horizontalalignment='center')
        ax.grid(True, linewidth=1, linestyle='--')
        ax.spines['polar'].set_linestyle('--')  # Dashed line style
        ax.spines['polar'].set_linewidth(0.5)     # Line width
        ax.spines['polar'].set_color('gray')    
        ax.spines['polar'].set_alpha(0.5)
        ax.yaxis.get_gridlines()[-1].set_visible(False)

    #fig.legend(loc='upper center', frameon=False, ncol=4)

fig, ax = plt.subplots(figsize=(1.5*FIG_WIDTH, 1.5*FIG_HEIGHT), subplot_kw={'projection': 'polar'})
plot_radar(pd.DataFrame(res_df_dict), ax)
fig.tight_layout(rect=[0, 0, 1, 0.99], h_pad=0.0)#, w_pad=0.0)
fig.legend(loc='upper center', frameon=False, ncol=3, fontsize=25, bbox_to_anchor=(0.5, 1.015), columnspacing=0.8)


this_file_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(this_file_dir, 'save_figs', f'figure_1_radar_{plot_name}.png'))
plt.savefig(os.path.join(this_file_dir, 'save_figs', f'figure_1_radar_{plot_name}.pdf'), format='pdf')
exit()
