
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind
# this_file_dir = 

home_directory = os.getenv('HOME') + '/'
if 'wxdeng' in home_directory:
    home_directory =  home_directory + '/oph/'

this_file_dir = home_directory + 'retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))

def calculate_quartiles_and_bounds(data):
    import numpy as np
    
    # 确保数据被排序
    data_sorted = np.sort(data)
    
    # 计算Q1和Q3
    Q1 = np.percentile(data_sorted, 25)
    Q3 = np.percentile(data_sorted, 75)
    
    # 计算IQR
    IQR = Q3 - Q1
    
    # 计算下界和上界
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return Q1, Q3, lower_bound, upper_bound

def find_min_max_indices(data):
    # 确保数据非空且为列表类型
    if not data or not isinstance(data, list):
        return None, None

    # 获取最小值和最大值的索引
    min_index = data.index(min(data))
    max_index = data.index(max(data))

    return min_index, max_index

# -----------------DATASET SETTINGS-----------------
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

RATIO_SETTING = [0.1 * i for i in range(1, 9)]

OIMHS_DATASET_DEVICE_DICT = {f'ratio_{ratio:.1f}':f'ratio_{ratio:.1f}' for ratio in RATIO_SETTING}
print("Inhouse dataset device dict: ", OIMHS_DATASET_DEVICE_DICT)
# {
#     "Ratio": 'ratio',
#     'Triton': 'triton',
#     "Maestro2": 'maes',
# }
OIMHS_DEVICE_PLOT_NAME = OIMHS_DATASET_DEVICE_DICT
# {
#     "Spectralis": "Heidelberg Spectralis",
#     'Triton': 'Topcon Triton',
#     "Maestro2": 'Topcon Maestro2',
# }

OIMHS_OPH_DATASET_EVAL_SETTING = dict([(key, ["default"]) for key in OIMHS_DATASET_DEVICE_DICT.keys()])
print("Inhouse dataset eval setting: ", OIMHS_OPH_DATASET_EVAL_SETTING)

# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
BASELINE = ["MAE-joint", "retfound"] #, "MAE2D"] #"linear_probe", "unlock_top"]
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
# SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}
SETTING_DICT = {"default": "runs_5_frames_18"} #, "lock_part": "unlock_part", "linear_probe": "linear_probe"}

# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "retfound": ["3D", "2D"],
    # "MAE2D": ["3D"],
    # "linear_probe": ["3D"],
    # "unlock_top": ["3D"],

} 

EXPR_DEFAULT_NAME_DICT = {
    "retfound 2D": ["outputs_ft_st_0622_moresize_ckpt_flash_attn", "retfound"],
    # "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
    "retfound 3D": ["outputs_ft_st_0622_moresize_ckpt_flash_attn", "retfound"],
    # "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
    # "MAE2D 3D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc",  "mae2d"],
    "MAE-joint 3D": ["outputs_ft_st_0622_moresize_ckpt_flash_attn", "nodrop"],
}

# Exteneded baseline methods with dimensionality and the plotting method name
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OctCube",
    "retfound 3D": "RETFound 3D",
    # "from_scratch 3D": "From scratch 3D",
    "retfound 2D": "RETFound 2D",
    # "MAE2D 3D": "2D MAE as 3D"
    # "from_scratch 2D": "From scratch 2D",
}

# -----------------MISC SUFFIX SETTINGS-----------------
# Miscellaneous suffix dictionary for the output folder
MISC_SUFFIX_DICT = {
    # ("glaucoma", "fewshot", "outputs_ft", "2D"): "new_correct_visit",
    # ("glaucoma", "fewshot", "outputs_ft", "3D"): "correct_visit",
    # ("glaucoma", "default", "outputs_ft", "2D"): "new_correct_visit",
    # ("glaucoma", "default", "outputs_ft", "3D"): "correct_visit",
    # ("glaucoma", "fewshot", "outputs_ft_st", "3D"): "correct_visit",
    # ("glaucoma", "default", "outputs_ft_st", "3D"): "correct_visit",
    # ("duke14", "fewshot", "outputs_ft", "2D"): "effective_fold",
    # ("duke14", "fewshot", "outputs_ft", "3D"): "effective_fold",
    # ("duke14", "fewshot", "outputs_ft_st", "3D"): "effective_fold",
    # ("hcms", "fewshot", "outputs_ft", "3D"): "correct_18",
    # ("hcms", "default", "outputs_ft", "3D"): "correct_18",
    # ("hcms", "fewshot", "outputs_ft", "2D"): "correct",
    # ("hcms", "default", "outputs_ft", "2D"): "correct",
    # ("hcms", "default", "outputs_ft_st", "3D"): "correct_18",
    # ("hcms", "fewshot", "outputs_ft_st", "3D"): "correct_18",
    # ("oimhs", "fewshot", "outputs_ft", "3D"): "correct_15",
    # ("oimhs", "default", "outputs_ft", "3D"): "correct_15",
    # ("oimhs", "fewshot", "outputs_ft", "2D"): "correct",
    # ("oimhs", "default", "outputs_ft", "2D"): "correct",
    # ("oimhs", "fewshot", "outputs_ft_st", "3D"): "correct_15",
    # ("oimhs", "default", "outputs_ft_st", "3D"): "correct_15",
    # ("maes", "default", "retfound", "3D"): "normalize",
    # ('maes', 'default', 'retfound', '3D'): "reproduce_10folds",
    # ('maes', 'default', 'MAE-joint', '3D'): "reproduce_10folds_new",
    # ('maes', 'default', 'retfound', '2D'): "50",
    # ('spec', 'default', 'retfound', '2D'): "50",
    # ('triton', 'default', 'retfound', '2D'): "50",
    # ("maes", "default", "MAE-joint", "3D"): ""
    # ("triton", "default", "retfound", "3D"): "normalize"
    ("ratio_0.8", "default", "MAE-joint", "3D"): "5fold",
    ("ratio_0.8", "default", "retfound", "3D"): "5fold",
    ("ratio_0.8", "default", "MAE-joint", "2D"): "5fold",
}

# Miscellaneous frame suffix dictionary for the output folder
MISC_FRAME_SUFFIX_DICT = {
    # ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    # ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
}
MISC_FILE_LOAD_DICT = {
    # ('spec', 'default', 'retfound', '3D'): ['fold_results_test_AUPRC', True],
    # ('spec', 'default', 'MAE-joint', '3D'): ['fold_results_test_AUPRC', True],
    # ('triton', 'default', 'mae2d', '3D'): ['fold_results_test_AUPRC', True],
} # True if need transpose
# -----------------END OF MISC SUFFIX SETTINGS-----------------

# ----------CUSTOMIZED COLOR SETTINGS-----------------
boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}

# ---------- END OF CUSTOMIZED COLOR SETTINGS-----------------


def oimhs_each_class_plot(ax, df_dict, metric='AUROC', min_y=0.55, max_y=0.8, rotation=30,plot_title=True):
    t = 'Pan 18-biomarker'
    if metric == 'AUROC':
        col2groups = {'PD-L1_auroc': r'$PD$-$L1$', 'TP53_auroc': r'$TP53$', 'LRP1B_auroc': r'$LRP1B$', 'KRAS_auroc': r'$KRAS$', 'APC_auroc': r'$APC$', 'KMT2D_auroc': r'$KMT2D$', 'FAT1_auroc': r'$FAT1$', 'SPTA1_auroc': r'$SPTA1$', 'ZFHX3_auroc': r'$ZFHX3$', 'KMT2C_auroc': r'$KMT2C$', 'EGFR_auroc': r'$EGFR$', 'ARID1A_auroc': r'$ARID1A$', 'PIK3CA_auroc': r'PIK3CA', 'PRKDC_auroc': r'$PRKDC$', 'NOTCH1_auroc': r'$NOTCH1$', 'ATM_auroc': r'$ATM$', 'KMT2A_auroc': r'$KMT2A$', 'ROS1_auroc': r'$ROS1$'}
    else:
        col2groups = {'PD-L1_auprc': r'$PD$-$L1$', 'TP53_auprc': r'$TP53$', 'LRP1B_auprc': r'$LRP1B$', 'KRAS_auprc': r'$KRAS$', 'APC_auprc': r'$APC$', 'KMT2D_auprc': r'$KMT2D$', 'FAT1_auprc': r'$FAT1$', 'SPTA1_auprc': r'$SPTA1$', 'ZFHX3_auprc': r'$ZFHX3$', 'KMT2C_auprc': r'$KMT2C$', 'EGFR_auprc': r'$EGFR$', 'ARID1A_auprc': r'$ARID1A$', 'PIK3CA_auprc': r'PIK3CA', 'PRKDC_auprc': r'$PRKDC$', 'NOTCH1_auprc': r'$NOTCH1$', 'ATM_auprc': r'$ATM$', 'KMT2A_auprc': r'$KMT2A$', 'ROS1_auprc': r'$ROS1$'}
    
    plot_cols = list(col2groups.keys())
    plot_methods = list(df_dict[t].keys())
    y = [np.mean(df_dict[t][plot_methods[0]][plot_col]) for plot_col in plot_cols]
    srt_idx = np.argsort(np.asarray(y))[::-1]
    plot_cols = [plot_cols[i] for i in srt_idx]
    plot_methods = np.asarray(plot_methods)
    
    n_groups = len(plot_cols)
    
    x = np.arange(n_groups)
    
    for i, m in enumerate(plot_methods):
        y = [np.mean(df_dict[t][m][plot_col]) for plot_col in plot_cols]
        if m == 'Prov-MSR-Path' or m == 'Prov-GigaPath':
            zorder = 3
        else:
            zorder = 2
        ax.scatter(x, y, s=20, label=m, color=COLORS[m], zorder=zorder, marker=MARKERS[m])
    format_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([col2groups[plot_col] for plot_col in plot_cols], rotation=rotation, ha='right')
    ax.set_ylabel(metric)
    ax.set_ylim(min_y, max_y)
    ax.legend(loc='upper right', frameon=False)
    
    if plot_title:
        ax.set_title(t)


def load_fold_results(file_path, transpose=False):
    """
    Load fold results from a given file path.
    
    Parameters:
    - file_path: str, the path to the file containing fold results.
    
    Returns:
    - np.ndarray, an array of the fold results.
    """
    data_floats = []  # To store the converted float values
    num_lines = 3
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('Fold results:'):
                # Process the line with fold results
                # print(line.strip())
                numbers = line.strip().split('[')[2].split(']')[0]  # Extract numbers between brackets
                # print(numbers)
                # print(len(numbers.split()))
                num_lines = len(numbers.split())
                data_floats.extend([float(num) for num in numbers.split()])
            elif 'Mean' in line or 'Std' in line:
                # Stop processing if we encounter "Mean" or "Std"
                break
            else:
                # Process any continuous lines with numbers
                numbers = line.strip().replace('[', '').replace(']', '')
                
                if numbers:  # If the line is not empty
                    data_floats.extend([float(num) for num in numbers.split()])
    
    # Convert the list of floats to a numpy array and reshape based on 3 columns
    results = np.array(data_floats).reshape(-1, num_lines)
    if transpose:
        results = results.T
    return results


def get_oimhs_results_json(out_dir, prefix='finetune_oimhs_', device="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    print('frame', frame, 'setting', setting, 'device', device, 'expr', expr)
    return out_dir + f"{prefix}{frame}_{device}{setting}{expr}/{fname}.{ext}"


def get_oimhs_task_and_setting_grouped_dict(results_dict):
    setting_list = SETTING_DICT.keys()
    task_list = OIMHS_DATASET_DEVICE_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = OIMHS_DATASET_DEVICE_DICT[task]
            available_setting = OIMHS_OPH_DATASET_EVAL_SETTING[task]
            if setting not in available_setting:
                continue
            grouped_dict[setting][task] = {}
            for baseline in BASELINE:

                for frame in FRAME_DICT.keys():
                    baseline_plus_frame = f"{baseline} {frame}"

                    # baseline_code = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame][0]
                    # if (task_code, setting_val, baseline_code, frame) in results_dict:
                    #     grouped_dict[setting][task][(baseline, frame)] = results_dict[(task_code, setting_val, baseline, frame)]
                    if (task, setting, baseline, frame) in results_dict:
                        grouped_dict[setting][task][(baseline, frame)] = results_dict[(task, setting, baseline, frame)]
    # print(grouped_dict)
    for setting in setting_list:
        for task in task_list:
            if task not in grouped_dict[setting]:
                print(f"Missing {task} in {setting}")
    return grouped_dict




def OIMHS_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
    '''
    plot the bar plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''

    df_dict = grouped_dict[setting_code]

    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
        # remove 'DUKE13'
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    print(plot_tasks, plot_methods)
    # exit()
    if plot_methods_name is None:
        plot_methods_name_key = [m[0] + ' ' + m[1] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods
    
    plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    plot_col_idx = plot_col_dict.index(plot_col)

    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend
    for i, plot_task in enumerate(plot_tasks):
        ax = axes[i]
        
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])

            handle = ax.bar((j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS_OIMHS[plot_methods_name_key[j]], zorder=3)
            if err_bar:
                y_std_err = np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist()))
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='gray', capsize=2, zorder=4)

            if y > best_y and j != 0:
                best_y = y
                compare_col = m
            if i == 0:  # Collect handle for legend only once per method across all tasks
                all_handles.append(handle)
                all_labels.append(plot_methods_name[j])

        y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.5])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.49]) 
        elif plot_col == 'bal_acc':
            y_min = np.min([y_min, 0.5])
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        print('y_max', y_max, plot_col, plot_task)
        print('y_min', y_min, plot_col, plot_task)  

        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        print(np.std(y_h), np.std(y_l))
        # print(calculate_quartiles_and_bounds(y_h))
        # print(calculate_quartiles_and_bounds(y_l))
        # print(find_min_max_indices(y_h))
        # print(find_min_max_indices(y_l))
        # idx_hh, idx_hl = find_min_max_indices(y_h)
        # idx_lh, idx_ll = find_min_max_indices(y_l)
        # # filter out the outliers
        # outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
        # y_h = [y_h[i] for i in range(len(y_h)) if i not in outlier_idx]
        # y_l = [y_l[i] for i in range(len(y_l)) if i not in outlier_idx]
        # Get the 1.5quantile of the data


        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        # t_stat, p_value = ttest_rel(y_h , y_l)
        t_stat, p_value = ttest_ind(y_h, y_l)
        # wilcoxon test
        # t_stat, p_value = wilcoxon(y_h*2, y_l*2, alternative='greater')
        print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks([])
        ax.set_xlabel(OIMHS_DEVICE_PLOT_NAME[plot_task], fontsize=12)
        # ax.set_xlabel(, fontsize=12)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=12)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        ax.tick_params(axis='y', labelsize=10)
    return all_handles, all_labels
    # add legend for the axes




def OIMHS_oph_tasks_scatterplot(fig, ax, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
    '''
    plot the bar plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''

    df_dict = grouped_dict[setting_code]

    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
        # remove 'DUKE13'
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    print(plot_tasks, plot_methods)
    # exit()
    if plot_methods_name is None:
        plot_methods_name_key = [m[0] + ' ' + m[1] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods
    
    plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    plot_col_idx = plot_col_dict.index(plot_col)

    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend
    for i, plot_task in enumerate(plot_tasks):
        # ax = axes[i]
        x = RATIO_SETTING[i]
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])

            # handle = ax.bar((j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS_OIMHS[plot_methods_name_key[j]], zorder=3)
            handle = ax.scatter(x, y, marker=MARKERS_OIMHS[plot_methods_name_key[j]], color=COLORS_OIMHS[plot_methods_name_key[j]], label=plot_methods_name[j], zorder=3)
            # if err_bar:
            #     y_std_err = np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
            #             np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist()))
            #     ax.errorbar(x, y, yerr=y_std_err, fmt='none', ecolor='gray', capsize=2, zorder=4)

            if y > best_y and j != 0:
                best_y = y
                compare_col = m
            if i == 0:  # Collect handle for legend only once per method across all tasks
                all_handles.append(handle)
                all_labels.append(plot_methods_name[j])

        y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.39])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.39]) 
        elif plot_col == 'bal_acc':
            y_min = np.min([y_min, 0.49])
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        print('y_max', y_max, plot_col, plot_task)
        print('y_min', y_min, plot_col, plot_task)  

        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        print(np.std(y_h), np.std(y_l))
        # print(calculate_quartiles_and_bounds(y_h))
        # print(calculate_quartiles_and_bounds(y_l))
        # print(find_min_max_indices(y_h))
        # print(find_min_max_indices(y_l))
        # idx_hh, idx_hl = find_min_max_indices(y_h)
        # idx_lh, idx_ll = find_min_max_indices(y_l)
        # # filter out the outliers
        # outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
        # y_h = [y_h[i] for i in range(len(y_h)) if i not in outlier_idx]
        # y_l = [y_l[i] for i in range(len(y_l)) if i not in outlier_idx]
        # Get the 1.5quantile of the data


        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        # t_stat, p_value = ttest_rel(y_h*2 , y_l*2)
        t_stat, p_value = ttest_ind(y_h*2, y_l*2)
        # wilcoxon test
        # t_stat, p_value = wilcoxon(y_h*2, y_l*2, alternative='greater')
        print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks(RATIO_SETTING)
        ax.tick_params(axis='x', labelsize=10)
        ax.set_xlabel('OIMHS', fontsize=12)
        # ax.set_xlabel(, fontsize=12)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=12)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            # ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            # ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            # ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text(x, line_y, stars, fontsize=12, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        ax.tick_params(axis='y', labelsize=10)
    return all_handles, all_labels
    # add legend for the axes

def OIMHS_oph_tasks_lineplot(fig, ax, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
    '''
    plot the line plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''

    df_dict = grouped_dict[setting_code]

    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
        # remove 'DUKE13'
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    print(plot_tasks, plot_methods)
    # exit()
    if plot_methods_name is None:
        plot_methods_name_key = [m[0] + ' ' + m[1] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods
    
    plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    color_alpha_dict = {'MAE-joint 3D': 0.5, 'retfound 3D': 0.2, 'retfound 2D': 0.2}
    assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    plot_col_idx = plot_col_dict.index(plot_col)

    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend
    all_method_val = []



    for j, m in enumerate(plot_methods):
        x_values = []
        y_values = []
        y_lower = []
        y_upper = []
        for i, plot_task in enumerate(plot_tasks):
            x = RATIO_SETTING[i]
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])
            x_values.append(x)
            y_values.append(y)

            y_std = np.std(df_dict[plot_task][m][:, plot_col_idx]) / np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx]))
            y_lower.append(y - y_std)
            y_upper.append(y + y_std)
            
        

            print('test:', x, plot_col, plot_task, m, y)
        
        handle, = ax.plot(x_values, y_values, marker=MARKERS_OIMHS[plot_methods_name_key[j]], color=COLORS_OIMHS[plot_methods_name_key[j]], label=plot_methods_name[j], zorder=3)
        ax.fill_between(x_values, y_lower, y_upper, color=COLORS_OIMHS[plot_methods_name_key[j]], alpha=color_alpha_dict[plot_methods_name_key[j]], zorder=2)

        all_handles.append(handle)
        all_labels.append(plot_methods_name[j])
        
        y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.39])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.39]) 
        elif plot_col == 'bal_acc':
            y_min = np.min([y_min, 0.49])
        y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        y_max = np.max([y_max, 0.93]) 
        print('y_max', y_max, plot_col, plot_task)
        print('y_min', y_min, plot_col, plot_task)  

        # y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        # y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        # print(np.std(y_h), np.std(y_l))

        # t_stat, p_value = ttest_ind(y_h, y_l)
        # print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks(RATIO_SETTING)
        ax.tick_params(axis='x', labelsize=30)
        ax.set_xlabel('Ratio of training data', fontsize=25)
        if j == 0:
            print('y_name', y_name)
            ax.set_ylabel(y_name, fontsize=25)
        
        delta_y = 0.01

        # stars = get_star_from_pvalue(p_value, star=True)
        # print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        # compare_idx = plot_methods.index(compare_col)
        # line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        
        format_ax(ax)
        # print('line_y', line_y, delta_y, line_y + 2*delta_y)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), y_max + 1*delta_y)
        ax.tick_params(axis='y', labelsize=30)
    for i, plot_task in enumerate(plot_tasks):

        x = RATIO_SETTING[i]
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):

            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])


            if y > best_y and j != 0:
                best_y = y
                compare_col = m

        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        print(np.std(y_h), np.std(y_l))
        # print(calculate_quartiles_and_bounds(y_h))
        # print(calculate_quartiles_and_bounds(y_l))
        # print(find_min_max_indices(y_h))
        # print(find_min_max_indices(y_l))
        # idx_hh, idx_hl = find_min_max_indices(y_h)
        # idx_lh, idx_ll = find_min_max_indices(y_l)
        # # filter out the outliers
        # outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
        # y_h = [y_h[i] for i in range(len(y_h)) if i not in outlier_idx]
        # y_l = [y_l[i] for i in range(len(y_l)) if i not in outlier_idx]
        # Get the 1.5quantile of the data


        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        # t_stat, p_value = ttest_rel(y_h*2 , y_l*2)
        t_stat, p_value = ttest_ind(y_h*2, y_l*2)
        # wilcoxon test
        # t_stat, p_value = wilcoxon(y_h*2, y_l*2, alternative='greater')
        print(compare_col, plot_methods_name[0], p_value)


        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            # ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            # ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            # ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text(x, line_y, stars, fontsize=35, ha='center', va='bottom')     

    return all_handles, all_labels

if __name__ == '__main__':
    results_dict = {}
    for device, device_code in OIMHS_DATASET_DEVICE_DICT.items():
        setting_list = OIMHS_OPH_DATASET_EVAL_SETTING[device] 
        for setting in setting_list:
            setting_val = SETTING_DICT[setting]
            for baseline in BASELINE:
                frame_list = INHOUSE_EVAL_FRAME[baseline]
                frame_value = [FRAME_DICT[frame] for frame in frame_list]

                for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                    frame = frame_list[i] # 3D, 2D
                    # print(name_dict)
                    baseline_plus_frame = f"{baseline} {frame}"
                    name_dict = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
                    out_folder = name_dict[0]
                    expr = name_dict[1]
                    suffix = ""
                    # print(out_folder)
                    # print((device_code, setting, baseline, frame))
                    # exit()
                    if (device_code, setting, baseline, frame) in MISC_SUFFIX_DICT:
                        suffix = MISC_SUFFIX_DICT[(device_code, setting, baseline, frame)]
                        # print(dataset, setting, out_folder, frame, suffix)
                    replace_fname = 'fold_results'
                    fname_list = ["fold_results"]
                    transpose = False
                    print((device_code, setting, baseline, frame))
                    if (device_code, setting, baseline, frame) in MISC_FILE_LOAD_DICT:
                        target_fname, transpose = MISC_FILE_LOAD_DICT[(device_code, setting, baseline, frame)]
                        fname_list = [target_fname]
                    for fname in fname_list:
                        for ext in ["txt"]:
                            if (device_code, setting, out_folder) in MISC_FRAME_SUFFIX_DICT:
                                frame_val = MISC_FRAME_SUFFIX_DICT[(device_code, setting, out_folder)][frame]
                            file_path = get_oimhs_results_json(this_file_dir + out_folder + '/', device=device_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)

                            print(f"Loading {file_path}")
                            try:
                                result = load_fold_results(file_path, transpose=transpose)
                                print(result)
                                results_dict[(device, setting, baseline, frame)] = result
                            except:
                                print(f"Error loading {file_path}")
                                replace_path = get_oimhs_results_json(this_file_dir + out_folder + '/', device=device_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext=ext)
                                print(f"Loading {replace_path}")
                                result = load_fold_results(replace_path, transpose=transpose)
                                print(result)
                                results_dict[(device, setting, baseline, frame)] = result
                                continue 
        print("\n")

    print(results_dict)
    grouped_dict = get_oimhs_task_and_setting_grouped_dict(results_dict)
    # exit()

    # Plot the figure
    # fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 1*FIG_HEIGHT), nrows=3, ncols=8)

    # results = {}
    # for task in TASKS:
        
    #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
    #     df_dict = {}
    #     for exp_code in EXP_CODE_DICT.keys():
    #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
    #     results[TASKS[task]] = df_dict

    # plot the subfigure a-e
    # OIMHS_oph_tasks_barplot(fig, axes[1, :], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUPRC')
    # # plot the subfigure f-j
    # OIMHS_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='default', plot_col='bal_acc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='BalAcc')
    # all_handles, all_labels = OIMHS_oph_tasks_barplot(fig, axes[0, :], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUROC')


    # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
    # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=4, fontsize=12, frameon=False)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.97)

    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_oimhs_a-f.pdf'), dpi=300)
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_oimhs_a-f.png'))





    # fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 1*FIG_HEIGHT), nrows=3, ncols=1)
    # OIMHS_oph_tasks_lineplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUPRC')
    # # plot the subfigure f-j
    # OIMHS_oph_tasks_lineplot(fig, axes[2], grouped_dict, setting_code='default', plot_col='bal_acc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='BalAcc')
    # all_handles, all_labels = OIMHS_oph_tasks_lineplot(fig, axes[0], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUROC')
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=4, fontsize=12, frameon=False)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.97)    
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_oimhs_a-f_line.pdf'), dpi=300)
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_oimhs_a-f_line.png'))


    # metric = 'auroc'
    # metric = 'auprc'
    # # metric = 'bal_acc'
    metric_name_dict = {'auroc': 'AUROC', 'auprc': 'AUPRC', 'acc': 'ACC', 'bal_acc': 'BalAcc'}
    # fig, axes = plt.subplots(figsize=(1.3*FIG_WIDTH, 1*FIG_HEIGHT), nrows=1, ncols=1)
    # # OIMHS_oph_tasks_lineplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUPRC')
    # # plot the subfigure f-j
    # # OIMHS_oph_tasks_lineplot(fig, axes[2], grouped_dict, setting_code='default', plot_col='bal_acc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='BalAcc')
    # all_handles, all_labels = OIMHS_oph_tasks_scatterplot(fig, axes, grouped_dict, setting_code='default', plot_col=metric, plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name=metric_name_dict[metric])
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=4, fontsize=30, frameon=False)
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.97)    
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', f'figure_oimhs_a-f_scatter_{metric}.pdf'), dpi=300)
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', f'figure_oimhs_a-f_scatter_{metric}.png'))



    fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 1*FIG_HEIGHT), nrows=1, ncols=2)
    # OIMHS_oph_tasks_lineplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='AUPRC')
    # plot the subfigure f-j
    # OIMHS_oph_tasks_lineplot(fig, axes[2], grouped_dict, setting_code='default', plot_col='bal_acc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name='BalAcc')
    metric = 'auroc'
    all_handles, all_labels = OIMHS_oph_tasks_lineplot(fig, axes[0], grouped_dict, setting_code='default', plot_col=metric, plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name=metric_name_dict[metric])
    metric = 'auprc'
    OIMHS_oph_tasks_lineplot(fig, axes[1], grouped_dict, setting_code='default', plot_col=metric, plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D')], y_name=metric_name_dict[metric])
    fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=25, frameon=False)
    
    # fig.subplots_adjust(top=0.97)    
    plt.savefig(os.path.join(save_file_dir, 'save_figs', f'figure_oimhs_a-f_line_{metric}_together.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', f'figure_oimhs_a-f_line_{metric}_together.png'))
    # fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=9)
    # INHOUSE_oph_tasks_barplot(fig, ax[0, :], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC')
    # INHOUSE_oph_tasks_barplot(fig, ax[1, :], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC')
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=9, fontsize=7, frameon=False)
    # fig.tight_layout()
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l.pdf'), dpi=300)
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l.png'))
    

    # mutation_18_biomarker_each_class_plot(ax, results, plot_title=False)

    # fig.tight_layout()
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l.pdf'), dpi=300)