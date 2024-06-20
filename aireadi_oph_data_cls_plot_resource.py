
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind
# this_file_dir = 

home_directory = os.getenv('HOME')
if 'wxdeng' in home_directory:
    home_directory =  home_directory + '/oph/'

this_file_dir = home_directory + 'retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))




# Baseline method and the corresponding avialable dimensional settings
AIREADI_EVAL_FT = {
    "MAE-joint": ["finetune", "lock_part", "linear_probe"],
    "retfound": ["finetune", "lock_part", "linear_probe"],
    # "MAE2D": ["3D"],
    # "linear_probe": ["3D"],
    # "unlock_top": ["3D"],

} 
AIREADI_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "retfound": ["3D"],
} 

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
    "Spectralis": 'spec',

    # 'Triton': 'triton',
    # "Maestro2": 'maes',
}
AIREADI_DEVICE_PLOT_NAME = {
    "Spectralis": "Heidelberg Spectralis",
    'Triton': 'Topcon Triton',
    "Maestro2": 'Topcon Maestro2',
}


AIREADI_FT_SETTING_DICT = {
    "finetune": "finetune",
    "lock_part": "lock_part",
    "linear_probe": "linear_probe",
}

AIREADI_OPH_DATASET_EVAL_SETTING = dict([(key, ["default"]) for key in AIREADI_DATASET_DEVICE_DICT.keys()])
print("Inhouse dataset eval setting: ", AIREADI_OPH_DATASET_EVAL_SETTING)
# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
BASELINE = ["MAE-joint", "retfound"] # , "MAE2D"] #"linear_probe", "unlock_top"]
FRAME_DICT = {"3D": "3D"} #, "2D": "2DCenter"}
# SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}
SETTING_DICT = {"default": "5fold_3d_256_0509"} #, "lock_part": "unlock_part", "linear_probe": "linear_probe"}

# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["finetune", "lock_part", "linear_probe"],
    "retfound": ["finetune", "lock_part", "linear_probe"],
    # "MAE2D": ["3D"],
    # "linear_probe": ["3D"],
    # "unlock_top": ["3D"],

} 
lock_number_groups = 6

EXPR_DEFAULT_NAME_DICT = {
    "MAE-joint finetune": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "nodrop"],
    "retfound finetune": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc", "retfound"],
    "retfound lock_part": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc_unlock_part", "retfound"],
    "MAE-joint lock_part": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc_unlock_part", "nodrop"],
    "MAE-joint linear_probe": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc_linear_probe", "nodrop"],
    "retfound linear_probe": ["outputs_ft_st_0612_2cls_01_ckpt_flash_attn_bal_acc_linear_probe", "retfound"],
}

# Exteneded baseline methods with dimensionality and the plotting method name
PLOT_METHODS_NAME = {
    # "MAE-joint 3D": "Ours model (3D MAE)",
    # "retfound 3D": "RETFound 3D",
    # "from_scratch 3D": "From scratch 3D",
    # "retfound 2D": "RETFound 2D",
    # "MAE2D 3D": "2D MAE as 3D"
    # "from_scratch 2D": "From scratch 2D",
    "MAE-joint finetune": "Ours model (3D MAE) - Fully finetune",
    "retfound finetune": "RETFound 3D - Fully finetune",
    "retfound lock_part": "RETFound 3D - last 6 layers",
    "MAE-joint lock_part": "Ours model (3D MAE) - last 6 layers",
    "MAE-joint linear_probe": "Ours model (3D MAE) - Linear probe",
    "retfound linear_probe": "RETFound 3D - Linear probe",


}

# -----------------MISC SUFFIX SETTINGS-----------------
# Miscellaneous suffix dictionary for the output folder
MISC_SUFFIX_DICT = {


    # ("maes", "default", "retfound", "3D"): "normalize"
    # ("triton", "default", "retfound", "3D"): "normalize"
}

# Miscellaneous frame suffix dictionary for the output folder
MISC_FRAME_SUFFIX_DICT = {
    # ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    # ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
}
MISC_FILE_LOAD_DICT = {
    ('spec', 'default', 'retfound', '3D', 'finetune'): ['fold_results_test_AUPRC', True],
    ('spec', 'default', 'MAE-joint', '3D', 'finetune'): ['fold_results_test_AUPRC', True],
    # ('triton', 'default', 'mae2d', '3D'): ['fold_results_test_AUPRC', True],
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
                for ft in AIREADI_EVAL_FT[baseline]:

                    for frame in FRAME_DICT.keys():
                        baseline_plus_frame = f"{baseline} {ft}"

                        # baseline_code = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame][0]
                        # if (task_code, setting_val, baseline_code, frame) in results_dict:
                        #     grouped_dict[setting][task][(baseline, frame)] = results_dict[(task_code, setting_val, baseline, frame)]
                        if (task, setting, baseline, frame, ft) in results_dict:
                            grouped_dict[setting][task][(baseline, frame, ft)] = results_dict[(task, setting, baseline, frame, ft)]
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
        plot_methods_name_key = [m[0] + ' ' + m[2] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods
    # exit()
    plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    plot_col_idx = plot_col_dict.index(plot_col)

    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend
    print(len(plot_tasks))
    for i, plot_task in enumerate(plot_tasks):
        if len(plot_tasks) > 1:
            ax = axes[i]
        else:
            ax = axes
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])

            handle = ax.bar((j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS_AIREADI_compute[plot_methods_name_key[j]], zorder=3)
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
            y_min = np.min([y_min, 0.52])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.52]) 
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        print('y_max', y_max)

        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        
        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        t_stat, p_value = ttest_rel(y_h * 2 , y_l * 2)
        # print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks([])
        ax.set_xlabel(AIREADI_DEVICE_PLOT_NAME[plot_task], fontsize=12)
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
        ax.tick_params(axis='y', labelsize=12)
    return all_handles, all_labels
    # add legend for the axes
    
def AIREADI_oph_memory_barplot(fig, axes, grouped_dict, memory_usage_dict, setting_code='default', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='GPU memory usage', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
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
        plot_methods_name_key = [m[0] + ' ' + m[2] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods
    # exit()
    # plot_col_dict = ['auroc', 'acc', 'auprc', 'bal_acc']
    # assert plot_col in plot_col_dict, f'plot_col should be one of {plot_col_dict}'
    # plot_col_idx = plot_col_dict.index(plot_col)

    
    # n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    all_handles = []  # List to store all handles for legend
    all_labels = []   # List to store all labels for legend
    print(len(plot_tasks))
    for i, plot_task in enumerate(plot_tasks):
        if len(plot_tasks) > 1:
            ax = axes[i]
        else:
            ax = axes
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            # y = np.mean(df_dict[plot_task][m][:, plot_col_idx])
            y = memory_usage_dict[m]

            handle = ax.bar((j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS_AIREADI_compute[plot_methods_name_key[j]], zorder=3)
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


        # y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        # if plot_col == 'auroc':
        #     y_min = np.min([y_min, 0.52])
        # elif plot_col == 'auprc':
        #     y_min = np.min([y_min, 0.52]) 
        # # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        # y_max = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
        #                 np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        y_min = 3
        y_max = 70
        print('y_max', y_max)

        # y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        # y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        
        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        # t_stat, p_value = ttest_rel(y_h * 2 , y_l * 2)
        # print(compare_col, plot_methods_name[0], p_value)

        ax.set_xticks([])
        ax.set_xlabel(AIREADI_DEVICE_PLOT_NAME[plot_task], fontsize=12)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=12)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        # delta_y = 0.01

        # stars = get_star_from_pvalue(p_value, star=True)
        # print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        # compare_idx = plot_methods.index(compare_col)
        # line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        # x1 = width
        # x2 = (compare_idx + 1)*width
        
        # if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
        #     ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        #     ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        #     ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        #     ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        format_ax(ax)
        # print('line_y', line_y, delta_y, line_y + 2*delta_y)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), y_max)
        ax.set_yticks([4, 8, 12, 16, 24, 32, 40, 48, 56, 64])
        ax.tick_params(axis='y', labelsize=12)
    return all_handles, all_labels
    # add legend for the axes
 

if __name__ == '__main__':
    results_dict = {}
    for device, device_code in AIREADI_DATASET_DEVICE_DICT.items():
        setting_list = AIREADI_OPH_DATASET_EVAL_SETTING[device] 
        for setting in setting_list:
            setting_val = SETTING_DICT[setting]
            for baseline in BASELINE:
                ft_list = AIREADI_EVAL_FT[baseline]
                ft_value = [AIREADI_FT_SETTING_DICT[frame] for frame in ft_list]
                frame_list = AIREADI_EVAL_FRAME[baseline]
                frame_value = [FRAME_DICT[frame] for frame in frame_list]

                for t, ft_val in enumerate(ft_value): # finetune, lock_part, linear_probe
                    ft = ft_list[t]
                    for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                        frame = frame_list[i] # 3D, 2D
                        # print(name_dict)
                        baseline_plus_frame = f"{baseline} {ft}"
                        name_dict = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
                        out_folder = name_dict[0]
                        expr = name_dict[1]
                        suffix = f'unlock_{lock_number_groups}' if ft == 'lock_part' else ""
                        # print(out_folder)
                        if (device_code, setting, baseline, frame) in MISC_SUFFIX_DICT:
                            suffix = MISC_SUFFIX_DICT[(device_code, setting, baseline, frame)]
                            # print(dataset, setting, out_folder, frame, suffix)
                        replace_fname = 'fold_results'
                        fname_list = ["fold_results"]
                        transpose = False
                        print((device_code, setting, baseline, frame, ft))
                        if (device_code, setting, baseline, frame, ft) in MISC_FILE_LOAD_DICT:
                            target_fname, transpose = MISC_FILE_LOAD_DICT[(device_code, setting, baseline, frame, ft)]
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
                                    results_dict[(device, setting, baseline, frame, ft)] = result
                                except:
                                    print(f"Error loading {file_path}")
                                    replace_path = get_aireadi_results_json(this_file_dir + out_folder + '/', device=device_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext=ext)
                                    print(f"Loading {replace_path}")
                                    result = load_fold_results(replace_path, transpose=transpose)
                                    print(result)
                                    results_dict[(device, setting, baseline, frame, ft)] = result
                                    continue 
        print("\n")

    print(results_dict)
    grouped_dict = get_aireadi_task_and_setting_grouped_dict(results_dict)
    print(grouped_dict)
    # exit()

    gpu_memory_dict = {
        ('MAE-joint', '3D', 'finetune'): 24.98,
        ('retfound', '3D', 'finetune'): 61.92,
        ('MAE-joint', '3D', 'lock_part'): 9.4,
        ('retfound', '3D', 'lock_part'): 19.1,
        ('MAE-joint', '3D', 'linear_probe'): 4.7,
        ('retfound', '3D', 'linear_probe'): 9.5,
    }
    # Plot the figure
    fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 1*FIG_HEIGHT), nrows=1, ncols=3)

    # results = {}
    # for task in TASKS:
        
    #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
    #     df_dict = {}
    #     for exp_code in EXP_CODE_DICT.keys():
    #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
    #     results[TASKS[task]] = df_dict

    # plot the subfigure a-e
    AIREADI_oph_tasks_barplot(fig, axes[0], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods= [('MAE-joint', '3D', 'finetune'), ('retfound', '3D', 'finetune'), ('MAE-joint', '3D', 'lock_part'),  ('retfound', '3D', 'lock_part'), ('MAE-joint', '3D', 'linear_probe'),  ('retfound', '3D', 'linear_probe')], y_name='AUPRC')
    # plot the subfigure f-j
    all_handles, all_labels = AIREADI_oph_tasks_barplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[('MAE-joint', '3D', 'finetune'), ('retfound', '3D', 'finetune'), ('MAE-joint', '3D', 'lock_part'),  ('retfound', '3D', 'lock_part'), ('MAE-joint', '3D', 'linear_probe'),  ('retfound', '3D', 'linear_probe')], y_name='AUROC')
    AIREADI_oph_memory_barplot(fig, axes[2], grouped_dict, gpu_memory_dict, setting_code='default', plot_tasks=[], plot_methods=[('MAE-joint', '3D', 'finetune'), ('retfound', '3D', 'finetune'), ('MAE-joint', '3D', 'lock_part'),  ('retfound', '3D', 'lock_part'), ('MAE-joint', '3D', 'linear_probe'),  ('retfound', '3D', 'linear_probe')], y_name='GPU memory usage', err_bar=False)
    # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
    # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=12, frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_4g-i.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_4g-i.png'))

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