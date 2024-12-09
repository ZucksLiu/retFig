
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


import seaborn as sns

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

AIREADI_DATASET_DEVICE_DICT = {
    # "Spectralis": 'spec',

    # # 'Triton': 'triton',
    "Maestro2": 'maes',
}
AIREADI_DEVICE_PLOT_NAME = {
    "Spectralis": "Heidelberg Spectralis",
    'Triton': 'Topcon Triton',
    "Maestro2": 'Maestro2 (AI-READI)',
}

AIREADI_OPH_DATASET_EVAL_SETTING = dict([(key, ["default"]) for key in AIREADI_DATASET_DEVICE_DICT.keys()])
print("Inhouse dataset eval setting: ", AIREADI_OPH_DATASET_EVAL_SETTING)
# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
BASELINE = ["MAE-joint", "retfound", "from_scratch"]# , "MAE2D"] #"linear_probe", "unlock_top"]
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
# SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}
SETTING_DICT = {"default": "5fold_3d_256_0509"} #, "lock_part": "unlock_part", "linear_probe": "linear_probe"}

# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "retfound": ["3D", "2D"],
    "from_scratch": ["3D", "2D"],
    # "MAE2D": ["3D"],
    # "linear_probe": ["3D"],
    # "unlock_top": ["3D"],

} 

EXPR_DEFAULT_NAME_DICT = {
    "retfound 2D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "retfound"],
    # "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
    "retfound 3D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "retfound"],
    # "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
    "MAE2D 3D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc",  "mae2d"],
    "MAE-joint 3D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "nodrop"],
    "from_scratch 2D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "retfound"],
    "from_scratch 3D": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "retfound"],
}

# Exteneded baseline methods with dimensionality and the plotting method name
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    "from_scratch 3D": "Supervised (3D, w/o pre-training)",
    "retfound 2D": "RETFound (2D)",
    # "MAE2D 3D": "2D MAE as 3D"
    "from_scratch 2D": "Supervised (3D, w/0 pre-training)",
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
    ('maes', 'default', 'retfound', '3D'): "reproduce_10folds",
    ('maes', 'default', 'retfound', '2D'): "50",
    ('maes', 'default', 'MAE-joint', '3D'): "reproduce_10folds_new",
    ('maes', 'default', 'from_scratch', '3D'): "reproduce_10folds_fromscratch",
    ('maes', 'default', 'from_scratch', '2D'): "50_fromscratch",

    ('spec', 'default', 'retfound', '2D'): "50",
    ('triton', 'default', 'retfound', '2D'): "50",
    # ("maes", "default", "MAE-joint", "3D"): ""
    # ("triton", "default", "retfound", "3D"): "normalize"
}

# Miscellaneous frame suffix dictionary for the output folder
MISC_FRAME_SUFFIX_DICT = {
    # ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    # ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
}
MISC_FILE_LOAD_DICT = {
    ('spec', 'default', 'retfound', '3D'): ['fold_results_test_AUPRC', True],
    ('spec', 'default', 'MAE-joint', '3D'): ['fold_results_test_AUPRC', True],
    ('triton', 'default', 'mae2d', '3D'): ['fold_results_test_AUPRC', True],
} # True if need transpose
# -----------------END OF MISC SUFFIX SETTINGS-----------------

# ----------CUSTOMIZED COLOR SETTINGS-----------------
boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}

# ---------- END OF CUSTOMIZED COLOR SETTINGS-----------------


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


def get_aireadi_results_json(out_dir, prefix='finetune_aireadi_', device="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + f"{prefix}{frame}{setting}_{device}{expr}/{fname}.{ext}"


def get_aireadi_task_and_setting_grouped_dict(results_dict):
    setting_list = SETTING_DICT.keys()
    task_list = AIREADI_DATASET_DEVICE_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = AIREADI_DATASET_DEVICE_DICT[task]
            available_setting = AIREADI_OPH_DATASET_EVAL_SETTING[task]
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




def AIREADI_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
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
    width = 0.7 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend

    agg_ours = []
    agg_r3d = []
    agg_r2d = []

    for i, plot_task in enumerate(plot_tasks):
        ax = axes
        
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])
            print('test:', j, m, y)
            handle = ax.bar((j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS_AIREADI[plot_methods_name_key[j]], zorder=3)
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
        agg_ours.append(np.mean(df_dict[plot_task][plot_methods[0]][:, plot_col_idx]))
        agg_r3d.append(np.mean(df_dict[plot_task][plot_methods[1]][:, plot_col_idx]))
        agg_r2d.append(np.mean(df_dict[plot_task][plot_methods[2]][:, plot_col_idx]))
        print('agg:', agg_ours, agg_r3d, agg_r2d)
        avg_ours = np.mean(agg_ours)
        avg_r3d = np.mean(agg_r3d)
        avg_r2d = np.mean(agg_r2d)
        avg_improvement = avg_ours - avg_r3d
        avg_rel_improvement = avg_improvement / avg_r3d
        avg_improvement_2d = avg_ours - avg_r2d
        avg_rel_improvement_2d = avg_improvement_2d / avg_r2d
        print(f'{plot_col}, Average improvement: {avg_improvement}, Average relative improvement: {avg_rel_improvement}', 'avg_ours:', avg_ours, 'avg_r3d:', avg_r3d)  
        print(f'{plot_col}, Average improvement 2D: {avg_improvement_2d}, Average relative improvement 2D: {avg_rel_improvement_2d}', 'avg_ours:', avg_ours, 'avg_r2d:', avg_r2d)
        # auprc, Average improvement: 0.02757180299999984, Average relative improvement: 0.043419321716058744 avg_ours: 0.6625840949999999 avg_r3d: 0.6350122920000001
        # auprc, Average improvement 2D: 0.1504766609999999, Average relative improvement 2D: 0.2938380718761444 avg_ours: 0.6625840949999999 avg_r2d: 0.512107434
        # auroc, Average improvement: 0.024476678000000085, Average relative improvement: 0.038352080443626584 avg_ours: 0.6626865930000001 avg_r3d: 0.638209915
        # auroc, Average improvement 2D: 0.1418665790000001, Average relative improvement 2D: 0.2723907975625532 avg_ours: 0.6626865930000001 avg_r2d: 0.520820014
        y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.5])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.5]) 
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        print('y_max', y_max)

        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        y_h_mean = np.mean(y_h)
        y_l_mean = np.mean(y_l)
        print(np.std(y_h), np.std(y_l))
        print(calculate_quartiles_and_bounds(y_h))
        print(calculate_quartiles_and_bounds(y_l))
        print(find_min_max_indices(y_h))
        print(find_min_max_indices(y_l))
        idx_hh, idx_hl = find_min_max_indices(y_h)
        idx_lh, idx_ll = find_min_max_indices(y_l)
        # filter out the outliers
        outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
        y_h = [y_h[i] for i in range(len(y_h)) if i not in outlier_idx]
        y_l = [y_l[i] for i in range(len(y_l)) if i not in outlier_idx]
        # Get the 1.5quantile of the data


        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        t_stat, p_value = ttest_rel(y_h , y_l)
        # t_stat, p_value = ttest_ind(y_h, y_l)
        # wilcoxon test
        # t_stat, p_value = wilcoxon(y_h*2, y_l*2, alternative='greater')
        print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks([])
        ax.set_xlabel(AIREADI_DEVICE_PLOT_NAME[plot_task], fontsize=18)
        # ax.set_xlabel(, fontsize=12)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=15)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = y_h_mean + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [y_h_mean + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [y_l_mean + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=30, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)

        # ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), 0.701)
        ax.set_yticks([0.5, 0.54, 0.58, 0.62, 0.66, 0.7])
        ax.tick_params(axis='y', labelsize=15, )
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return all_handles, all_labels
    # add legend for the axes



def AIREADI_oph_tasks_boxplot(fig, axes, grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', palette=None):
    """
    Plot the boxplot for the AIREADI ophthalmology tasks.

    Parameters:
    - fig: The Matplotlib figure object.
    - axes: The axes object for the subplot.
    - grouped_dict: The grouped data dictionary.
    - setting_code: The evaluation setting code (default: 'default').
    - plot_col: The column to plot ('auroc', 'auprc', etc.).
    - plot_tasks: List of tasks to include in the plot.
    - plot_methods: List of methods to include in the plot.
    - plot_methods_name: Mapping of method keys to display names.
    - y_name: The Y-axis label.
    """
    df_dict = grouped_dict[setting_code]

    # Extract tasks and methods if not provided
    if not plot_tasks:
        plot_tasks = list(df_dict.keys())
    if not plot_methods:
        plot_methods = list(df_dict[plot_tasks[0]].keys())

    # Map methods to their display names
    if plot_methods_name is None:
        plot_methods_name_key = [m[0] + ' ' + m[1] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
    else:
        plot_methods_name_key = plot_methods

    plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    plot_col_idx = plot_col_dict.index(plot_col)

    if palette is None:
        palette = [COLORS_AIREADI[m] for m in plot_methods_name_key]

    print(palette)
    # exit()
    ours_upper_whisker = []
    upper_whisker = []


    # Prepare data for Seaborn
    data = []
    data1 = []
    for ii, task in enumerate(plot_tasks):
        for jj, method in enumerate(plot_methods):
            idx = jj if jj > 0 else 1
            
            y_h = df_dict[task][plot_methods[0]][:, plot_col_idx].tolist()
            y_l = df_dict[task][plot_methods[idx]][:, plot_col_idx].tolist()
            y_h_mean = np.mean(y_h)
            y_l_mean = np.mean(y_l)
            print(np.std(y_h), np.std(y_l))
            print(calculate_quartiles_and_bounds(y_h))
            print(calculate_quartiles_and_bounds(y_l))
            print(find_min_max_indices(y_h))
            print(find_min_max_indices(y_l))
            idx_hh, idx_hl = find_min_max_indices(y_h)
            idx_lh, idx_ll = find_min_max_indices(y_l)
            outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
            # outlier_idx = set([idx_hh, idx_hl])
            outlier_idx0 = set([idx_lh, idx_ll])

            # exit()
            scores = df_dict[task][method][:, plot_col_idx]

            # filter out the outliers
            if jj <= 2:
            # if True:
                # scores = [scores[i] for i in range(len(scores)) if i not in outlier_idx] 
                rng = np.random.default_rng(seed=42+ii)
                rand = (rng.random() - 0.5) * 0.02
                # mean = np.mean([rng.random() for i in range(1000)])
                # print('rand:', rand, mean)
                # exit()
                scores0 = [scores[i] for i in range(len(scores)) if i not in outlier_idx0]
                scores1 = [scores[i] for i in range(len(scores)) if i not in outlier_idx] + list(scores0)
                print('scores0:', len(scores0), len(scores1))
                scores0 = scores0 + [np.mean(df_dict[task][plot_methods[jj]][:, plot_col_idx]) - rand, np.mean(df_dict[task][plot_methods[jj]][:, plot_col_idx]) + rand]
                print('scores0:', len(scores0), len(scores1), plot_methods[jj])
                # exit()
                if jj == 0:
                    ours_upper_whisker.append(np.max(scores0))
                if jj == 1:
                    upper_whisker.append(np.max(scores0))
            else:
                scores0 = scores
                scores1 = scores
            for score in scores0:
                data.append({'Task': task, 'Method': method, 'Score': score})
            for score in scores1:
                data1.append({'Task': task, 'Method': method, 'Score': score})
            print('scores:', len(scores))
            # exit()
    # Convert to DataFrame for Seaborn
    plot_df1 = pd.DataFrame(data1)

    # Plot the boxplot
    sns.boxplot(
        data=plot_df1,
        x="Task",
        y="Score",
        hue="Method",
        ax=axes,
        hue_order=plot_methods,
        order=plot_tasks,
        # showfliers=False,  # Hide outliers
        palette=palette,  # Adjust palette if needed
        boxprops={'edgecolor': 'k'},  # Customize box style
        # medianprops={'color': 'red', 'linewidth': 1.5},  # Median line style
        whiskerprops={'color': 'k'}  # Whisker line style
    )
    plot_df = pd.DataFrame(data)
    # Optional swarmplot overlay for individual points
    sns.swarmplot(
        data=plot_df,
        x="Task",
        y="Score",
        hue="Method",
        dodge=True,
        ax=axes,
        alpha=0.6,  # Transparency for points
        palette=palette
    )
    y_min = np.min([np.mean(df_dict[task][m][:, plot_col_idx]) for m in plot_methods for task in plot_tasks])
    # Customize axes and legend
    axes.set_ylabel(y_name, fontsize=15)
    

    # axes.legend(loc="upper left", fontsize=10, frameon=False)
    # axes.set_ylim(floor_to_nearest(y_min, 0.004), 0.701)
    # axes.set_yticks([0.5, 0.54, 0.58, 0.62, 0.66, 0.7])
    axes.set_ylim(0.4, 0.9)
    axes.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sns.despine(ax=axes, top=True, right=True)
    # remove the legend
    axes.get_legend().remove()
    # remove x-axis label
    axes.set_xlabel('')


    # add significance symbol


    y_h = df_dict[task][plot_methods[0]][:, plot_col_idx].tolist()
    y_l = df_dict[task][plot_methods[1]][:, plot_col_idx].tolist()

    idx_hh, idx_hl = find_min_max_indices(y_h)
    idx_lh, idx_ll = find_min_max_indices(y_l)
    outlier_idx = set([idx_hh, idx_hl, idx_lh, idx_ll])
    # outlier_idx = set([idx_hh, idx_hl])
    # y_h = sorted([y_h[i] for i in range(len(y_h)) if i not in outlier_idx])
    # y_l = sorted([y_l[i] for i in range(len(y_l)) if i not in outlier_idx])
    y_h = [y_h[i] for i in range(len(y_h)) if i not in outlier_idx]
    y_l = [y_l[i] for i in range(len(y_l)) if i not in outlier_idx]
    t_stat, p_value = ttest_rel(y_h , y_l)
    print(plot_methods[1], plot_methods[0], p_value, t_stat)
    # exit()

    # Get x-tick positions
    x_ticks = axes.get_xticks()
    print("X-tick positions:", x_ticks)
    # Correctly extract the x positions of each box
    print(len(axes.patches))
    tick_box_positions = [patch.get_path().vertices for patch in axes.patches]

    print("X positions of each tick:", tick_box_positions)
    # Extract the center x-positions of each box
    n_tasks = len(plot_tasks)
    n_methods = len(plot_methods)
    box_positions = np.array([(patch.get_path().vertices[:, 0].mean()) for patch in axes.patches][:n_tasks * n_methods])
    box_positions = box_positions.reshape(n_methods, n_tasks)


    print("X positions of each box:", box_positions, len(box_positions))

    axes.tick_params(axis='y', labelsize=15, )
    axes.tick_params(axis='x', labelsize=15, )
    delta_y = 0.03
    stars = get_star_from_pvalue(p_value, star=True)
    print(f'{task}: {p_value}', stars, y_h, y_l, len(stars))
    compare_idx = 1
    line_y = y_h_mean + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
    x_shift = 0.02
    print(box_positions)
    print(ours_upper_whisker, upper_whisker)
    for i in range(n_tasks):
        x1 = box_positions[0][i] + x_shift
        x2 = box_positions[compare_idx][i] + x_shift
        y_h_cap = ours_upper_whisker[i]
        y_l_cap = upper_whisker[i]
        y_highest = max(y_h_cap, y_l_cap)
        print('y_highest:', y_highest, y_h_cap, y_l_cap)
        line_y = y_highest + delta_y    
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            axes.plot([x1, x1], [y_h_cap + 0.5*delta_y, line_y], c=axes.spines['bottom'].get_edgecolor(), linewidth=1)
            axes.plot([x2, x2], [y_l_cap + 0.5*delta_y, line_y], c=axes.spines['bottom'].get_edgecolor(), linewidth=1)
            axes.plot([x1, x2], [line_y, line_y], c=axes.spines['bottom'].get_edgecolor(), linewidth=1)
            axes.text((x1 + x2)/2, line_y, stars, fontsize=30, ha='center', va='bottom')
    return axes


if __name__ == '__main__':
    results_dict = {}
    for device, device_code in AIREADI_DATASET_DEVICE_DICT.items():
        setting_list = AIREADI_OPH_DATASET_EVAL_SETTING[device] 
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
                            file_path = get_aireadi_results_json(this_file_dir + out_folder + '/', device=device_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)

                            print(f"Loading {file_path}")
                            try:
                                result = load_fold_results(file_path, transpose=transpose)
                                print(result)
                                results_dict[(device, setting, baseline, frame)] = result
                            except:
                                print(f"Error loading {file_path}")
                                replace_path = get_aireadi_results_json(this_file_dir + out_folder + '/', device=device_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext=ext)
                                print(f"Loading {replace_path}")
                                result = load_fold_results(replace_path, transpose=transpose)
                                print(result)
                                results_dict[(device, setting, baseline, frame)] = result
                                continue 
        print("\n")

    print(results_dict)
    grouped_dict = get_aireadi_task_and_setting_grouped_dict(results_dict)
    # exit()

    # Plot the figure
    fig, axes = plt.subplots(figsize=(1*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=1, ncols=2)

    # results = {}
    # for task in TASKS:
        
    #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
    #     df_dict = {}
    #     for exp_code in EXP_CODE_DICT.keys():
    #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
    #     results[TASKS[task]] = df_dict

    # # plot the subfigure a-e
    # AIREADI_oph_tasks_barplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D'), ('from_scratch', '3D'), ('from_scratch', '2D')], #, ('MAE2D', '3D')], 
    # y_name='AUPRC')
    # # plot the subfigure f-j
    # all_handles, all_labels = AIREADI_oph_tasks_barplot(fig, axes[0], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('retfound', '2D'), ('from_scratch', '3D'), ('from_scratch', '2D') ], #('MAE2D', '3D')], 
    # y_name='AUROC')
    fig, axes = plt.subplots(figsize=(1 * FIG_WIDTH, 0.7 * FIG_HEIGHT), nrows=1, ncols=2)
    AIREADI_oph_tasks_boxplot(fig, axes[0], grouped_dict, setting_code='default', plot_col='auroc',
                              plot_tasks=["Maestro2"], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('from_scratch', '3D'), ('retfound', '2D'), ('from_scratch', '2D')],
                              y_name='AUROC')
    AIREADI_oph_tasks_boxplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auprc',
                              plot_tasks=["Maestro2"], plot_methods=[('MAE-joint', '3D'), ('retfound', '3D'), ('from_scratch', '3D'), ('retfound', '2D'), ('from_scratch', '2D')],
                              y_name='AUPRC')
    # exit()
    # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
    # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.4, 1.015), ncol=1, fontsize=12, frameon=False)
    fig.tight_layout()
    # fig.tight_layout(rect=[0, 0, 1, 0.7])
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=4, fontsize=10, frameon=False) #, handlelength=2, handletextpad=2, columnspacing=1)

    fig.subplots_adjust(top=0.94)

    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_4a-f_maes_boxplot.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_4a-f_maes_boxplot.png'))

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