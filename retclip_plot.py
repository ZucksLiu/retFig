
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))
retclip_exp_res = os.path.join(save_file_dir, 'retClip_exp_res.csv')
retclip_exp_res_df = pd.read_csv(retclip_exp_res)
print(retclip_exp_res_df)

PLOT_METHODS_NAME = {
    "MAE-joint": "Ours model (3D MAE)",
    "retFound 3D": "RETFound 3D",
    # "from_scratch 3D": "From scratch 3D",
    "retFound 2D": "RETFound 2D",
    # "from_scratch 2D": "From scratch 2D",
}


def plot_retclip_recall_and_mean_rank(axes, retclip_exp_res_df, prefix, col_names, reverse_plot=True):
    plot_method = retclip_exp_res_df['Method'].tolist()
    print(plot_method)
    full_col_names = [f'{prefix} {col_name}' for col_name in col_names]
    print(full_col_names)
    all_handles = []
    all_labels = []
    for i, col_name in enumerate(col_names):
        ax = axes[i]
        y = retclip_exp_res_df[full_col_names[i]].tolist()
        print(y)
        
        width = 0.8 / len(plot_method)
        if reverse_plot:
            y = y[::-1]
        for j in range(len(plot_method)):
            method = plot_method[::-1][j]
            method_label = PLOT_METHODS_NAME[method]
            handle = ax.bar((j + 1)*width, y[j], width, label=method_label, color=COLORS[method], zorder=3,)
            if i == 0:
                all_handles.append(handle)
                all_labels.append(method_label)
                print(all_labels, all_handles)
        ax.set_xticks([])
        ax.set_xlabel(capitalize_first_letter(col_name), fontsize=8)
        if i == 0:
            ax.set_ylabel(prefix.replace('_', ' '), fontsize=8)
        y_max = np.max(y)
        y_min = np.min(y)
        print('y_max', y_max, 'y_min', y_min)
        if col_name == 'mean rank':
            ax.set_ylim(0, y_max + 5)
        else:
            ax.set_ylim(floor_to_nearest(y_min, 0.004) - 0.1, y_max + 0.1)
        format_ax(ax)
    return all_handles, all_labels

    # # handle = ax.bar((j + 1)*width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name[j]], zorder=3)
    # exit()
    # return ax



def plot_retclip_exp_res(retclip_exp_res_df):
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
    first_row_prefix = 'OCT_to_IR'
    second_row_prefix = 'IR_to_OCT'
    col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
    all_handles, all_labels = plot_retclip_recall_and_mean_rank(ax[0, :], retclip_exp_res_df, first_row_prefix, col_names)
    plot_retclip_recall_and_mean_rank(ax[1, :], retclip_exp_res_df, second_row_prefix, col_names)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.012), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res.png'))
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res.pdf'), dpi=300)
    # Draw bar plot for each metric
    
plot_retclip_exp_res(retclip_exp_res_df)

exit()


# INHOUSE_OPH_DATASET_DICT = {
#     # "DUKE13": "duke13",
#     "MULTI_LABEL": "multi_label",
#     "POG": "BCLS_POG",
#     "DME": "BCLS_DME",
#     "AMD": "BCLS_AMD",
#     "ODR": "BCLS_ODR",
#     "PM": "BCLS_PM",
#     "CRO": "BCLS_CRO",
#     "VD": "BCLS_VD",
#     "RN": "BCLS_RN",
#     }

# BASELINE = ["MAE-joint", "retfound", "from_scratch"]

# INHOUSE_OPH_DATASET_EVAL_SETTING = dict([(key, ["fewshot", "default"]) for key in INHOUSE_OPH_DATASET_DICT.keys()])



# INHOUSE_EVAL_FRAME = {
#     "retfound": ["3D", "2D"],
#     "from_scratch": ["3D", "2D"],
#     "MAE-joint": ["3D"],
# } 
# print("Inhouse dataset eval setting: ", INHOUSE_OPH_DATASET_EVAL_SETTING)

# FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
# SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}

# EXPR_DEFAULT_NAME_DICT = {
#     "retfound 2D": ["outputs_ft_0529_ckpt_flash_attn", "retfound"],
#     "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
#     "retfound 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_retfound"],
#     "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
#     "MAE-joint 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_3d_256_0509"],
# }

# MISC_SUFFIX_DICT = {
#     # ("glaucoma", "fewshot", "outputs_ft", "2D"): "new_correct_visit",
#     # ("glaucoma", "fewshot", "outputs_ft", "3D"): "correct_visit",
#     # ("glaucoma", "default", "outputs_ft", "2D"): "new_correct_visit",
#     # ("glaucoma", "default", "outputs_ft", "3D"): "correct_visit",
#     # ("glaucoma", "fewshot", "outputs_ft_st", "3D"): "correct_visit",
#     # ("glaucoma", "default", "outputs_ft_st", "3D"): "correct_visit",
#     # ("duke14", "fewshot", "outputs_ft", "2D"): "effective_fold",
#     # ("duke14", "fewshot", "outputs_ft", "3D"): "effective_fold",
#     # ("duke14", "fewshot", "outputs_ft_st", "3D"): "effective_fold",
#     # ("hcms", "fewshot", "outputs_ft", "3D"): "correct_18",
#     # ("hcms", "default", "outputs_ft", "3D"): "correct_18",
#     # ("hcms", "fewshot", "outputs_ft", "2D"): "correct",
#     # ("hcms", "default", "outputs_ft", "2D"): "correct",
#     # ("hcms", "default", "outputs_ft_st", "3D"): "correct_18",
#     # ("hcms", "fewshot", "outputs_ft_st", "3D"): "correct_18",
#     # ("oimhs", "fewshot", "outputs_ft", "3D"): "correct_15",
#     # ("oimhs", "default", "outputs_ft", "3D"): "correct_15",
#     # ("oimhs", "fewshot", "outputs_ft", "2D"): "correct",
#     # ("oimhs", "default", "outputs_ft", "2D"): "correct",
#     # ("oimhs", "fewshot", "outputs_ft_st", "3D"): "correct_15",
#     # ("oimhs", "default", "outputs_ft_st", "3D"): "correct_15",
# }

# MISC_FRAME_SUFFIX_DICT = {
#     # ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
#     # ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
# }

# boxprops = {'edgecolor': 'k'}
# whiskerprops = {'color': 'k'}
# capprops = {'color': 'k'}
# medianprops = {'color': 'r', 'linewidth': 1.5}

# def load_fold_results(file_path):
#     """
#     Load fold results from a given file path.
    
#     Parameters:
#     - file_path: str, the path to the file containing fold results.
    
#     Returns:
#     - np.ndarray, an array of the fold results.
#     """
#     data_floats = []  # To store the converted float values
#     num_lines = 3
#     with open(file_path, 'r') as file:
#         for line in file:
#             if line.startswith('Fold results:'):
#                 # Process the line with fold results
#                 # print(line.strip())
#                 numbers = line.strip().split('[')[2].split(']')[0]  # Extract numbers between brackets
#                 # print(numbers)
#                 # print(len(numbers.split()))
#                 num_lines = len(numbers.split())
#                 data_floats.extend([float(num) for num in numbers.split()])
#             elif 'Mean' in line or 'Std' in line:
#                 # Stop processing if we encounter "Mean" or "Std"
#                 break
#             else:
#                 # Process any continuous lines with numbers
#                 numbers = line.strip().replace('[', '').replace(']', '')
                
#                 if numbers:  # If the line is not empty
#                     data_floats.extend([float(num) for num in numbers.split()])
    
#     # Convert the list of floats to a numpy array and reshape based on 3 columns
#     results = np.array(data_floats).reshape(-1, num_lines)
#     return results

# def get_results_json(out_dir, prefix='finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
#     if setting != "":
#         setting = f"_{setting}"
#     if expr != "":
#         expr = f"_{expr}"
#     if suffix != "":
#         expr = f"{expr}_{suffix}"
#         # print(expr, suffix)
#     return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"

# def get_inhouse_task_and_setting_grouped_dict(results_dict):
#     setting_list = SETTING_DICT.keys()
#     task_list = INHOUSE_OPH_DATASET_DICT.keys()
#     grouped_dict = {}
#     for setting in setting_list:
#         setting_val = SETTING_DICT[setting]
#         grouped_dict[setting] = {}
#         for task in task_list:
#             task_code = INHOUSE_OPH_DATASET_DICT[task]
#             available_setting = INHOUSE_OPH_DATASET_EVAL_SETTING[task]
#             if setting not in available_setting:
#                 continue
#             grouped_dict[setting][task] = {}
#             for baseline in BASELINE:

#                 for frame in FRAME_DICT.keys():
#                     baseline_plus_frame = f"{baseline} {frame}"

#                     # baseline_code = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame][0]
#                     # if (task_code, setting_val, baseline_code, frame) in results_dict:
#                     #     grouped_dict[setting][task][(baseline, frame)] = results_dict[(task_code, setting_val, baseline, frame)]
#                     if (task, setting, baseline, frame) in results_dict:
#                         grouped_dict[setting][task][(baseline, frame)] = results_dict[(task, setting, baseline, frame)]
#     # print(grouped_dict)
#     for setting in setting_list:
#         for task in task_list:
#             if task not in grouped_dict[setting]:
#                 print(f"Missing {task} in {setting}")
#     return grouped_dict




# def INHOUSE_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=False):
#     '''
#     plot the bar plot for the mutation 6 tasks
#     df_dict: results for the mutation 6 tasks and all comparison approaches
#     '''



#     df_dict = grouped_dict[setting_code]

#     if len(plot_tasks) == 0:
#         plot_tasks = list(df_dict.keys())
#         # remove 'DUKE13'
#     if len(plot_methods) == 0:
#         plot_methods = list(df_dict[plot_tasks[0]].keys())
#         plot_methods_name = [m[0] + ' ' + m[1] for m in plot_methods]
#         print(plot_methods_name)


#     plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
#     assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
#     plot_col_idx = plot_col_dict.index(plot_col)

#     # plot_tasks.remove(f'Pan ' + r'$EFGR$' + ' subvariants')
    
#     n_groups = len(plot_tasks)
#     n_methods = len(plot_methods)
#     width = 0.8 / n_methods # set the barplot width
#     all_handles = []  # List to store all handles for legend
#     all_labels = []   # List to store all labels for legend
#     for i, plot_task in enumerate(plot_tasks):
#         ax = axes[i]
        
#         best_y, compare_col = -np.inf, ''
#         for j, m in enumerate(plot_methods):
#             print(j, m)
#             y = np.mean(df_dict[plot_task][m][:, plot_col_idx])

#             handle = ax.bar((j + 1)*width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name[j]], zorder=3)
#             if err_bar:
#                 y_std_err = np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
#                         np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist()))
#                 ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)

#             if y > best_y and j != 0:
#                 best_y = y
#                 compare_col = m
#             if i == 0:  # Collect handle for legend only once per method across all tasks
#                 all_handles.append(handle)
#                 all_labels.append(plot_methods_name[j])

#         y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
#         if plot_col == 'auroc':
#             y_min = np.min([y_min, 0.5])
#         elif plot_col == 'auprc':
#             y_min = np.min([y_min, 0.4]) 
#         # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
#         y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
#                         np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])

#         y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
#         y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        
#         # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
#         t_stat, p_value = ttest_rel(y_h * 2 , y_l * 2)
#         # print(compare_col, plot_methods_name[0], p_value)

#         ax.set_xticks([])
#         ax.set_xlabel(plot_task)
#         if i == 0:
#             ax.set_ylabel(y_name)
#         #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
#         # add significance symbol
#         delta_y = 0.01

#         stars = get_star_from_pvalue(p_value, star=True)
#         print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
#         compare_idx = plot_methods.index(compare_col)
#         line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
#         x1 = width
#         x2 = (compare_idx + 1)*width
        
#         if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
#             ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
#             ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
#             ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
#             ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
#         format_ax(ax)
#         print('line_y', line_y, delta_y, line_y + 2*delta_y)
#         ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
#     return all_handles, all_labels
#     # add legend for the axes
    


# if __name__ == '__main__':
#     results_dict = {}
#     for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
#         setting_list = INHOUSE_OPH_DATASET_EVAL_SETTING[dataset] 
#         for setting in setting_list:
#             setting_val = SETTING_DICT[setting]
#             for baseline in BASELINE:
#                 frame_list = INHOUSE_EVAL_FRAME[baseline]
#                 frame_value = [FRAME_DICT[frame] for frame in frame_list]

#                 for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
#                     frame = frame_list[i] # 3D, 2D
#                     # print(name_dict)
#                     baseline_plus_frame = f"{baseline} {frame}"
#                     name_dict = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
#                     out_folder = name_dict[0]
#                     expr = name_dict[1]
#                     suffix = ""
#                     # print(out_folder)
#                     if (dataset_code, setting, out_folder, frame) in MISC_SUFFIX_DICT:
#                         suffix = MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
#                         # print(dataset, setting, out_folder, frame, suffix)
#                     replace_fname = 'fold_results_test_for_best_val'
#                     for fname in ["fold_results_test"]:
#                         for ext in ["txt"]:
#                             if (dataset_code, setting, out_folder) in MISC_FRAME_SUFFIX_DICT:
#                                 frame_val = MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
#                             file_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)

#                             # print(f"Loading {file_path}")
#                             try:
#                                 result = load_fold_results(file_path)
#                                 print(result)
#                                 results_dict[(dataset, setting, baseline, frame)] = result
#                             except:
#                                 print(f"Error loading {file_path}")
#                                 replace_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext=ext)
#                                 print(f"Loading {replace_path}")
#                                 result = load_fold_results(replace_path)
#                                 print(result)
#                                 results_dict[(dataset, setting, baseline, frame)] = result
#                                 continue 
#         print("\n")
#     # exit()
#     print(results_dict)
#     grouped_dict = get_inhouse_task_and_setting_grouped_dict(results_dict)
#     # exit()

#     # Plot the figure
#     fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=9)

#     # results = {}
#     # for task in TASKS:
        
#     #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
#     #     df_dict = {}
#     #     for exp_code in EXP_CODE_DICT.keys():
#     #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
#     #     results[TASKS[task]] = df_dict

#     # plot the subfigure a-e
#     INHOUSE_oph_tasks_barplot(fig, axes[0, :], grouped_dict, setting_code='fewshot', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC')
#     # plot the subfigure f-j
#     all_handles, all_labels = INHOUSE_oph_tasks_barplot(fig, axes[1, :], grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC')
#     # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
#     # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
#     # plot the subfigure k
#     # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
#     fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=9, fontsize=7, frameon=False)
#     fig.tight_layout()

#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3a-k.pdf'), dpi=300)
#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3a-k.png'))

#     fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=9)
#     INHOUSE_oph_tasks_barplot(fig, ax[0, :], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC')
#     INHOUSE_oph_tasks_barplot(fig, ax[1, :], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC')
#     fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=9, fontsize=7, frameon=False)
#     fig.tight_layout()
#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l.pdf'), dpi=300)
#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l.png'))
    

#     # mutation_18_biomarker_each_class_plot(ax, results, plot_title=False)

#     # fig.tight_layout()
#     # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l.pdf'), dpi=300)