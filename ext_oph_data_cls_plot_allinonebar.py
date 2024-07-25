
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))

FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}

PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    "from_scratch 3D": "Supervised (3D, w/o pre-training)",
    "retfound 2D": "RETFound (2D)",
    "from_scratch 2D": "Supervised (2D, w/o pre-training)",
}

boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}
# ---------- END OF CUSTOMIZED COLOR SETTINGS-----------------

# -----------------DATASET SETTINGS-----------------
# External Ophthalmology datasets mapping
EXT_OPH_DATASET_DICT = {
    # "DUKE13": "duke13",
    "DUKE14": "duke14",
    "GLAUCOMA": "glaucoma",
    "UMN": "umn",
    "OIMHS": "oimhs",
    "HCMS": "hcms",
    } 

# External Ophthalmology datasets and the corresponding available experimental settings
EXT_OPH_DATASET_EVAL_SETTING = {
    # "DUKE13": ["fewshot"],
    "DUKE14": ["fewshot"],
    "GLAUCOMA":["fewshot", "default"],
    "UMN": ["fewshot"],
    "OIMHS": ["fewshot", "default"],
    "HCMS": ["fewshot", "default"],
}
# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
# Baseline methods
BASELINE = ["MAE-joint", "retfound", "from_scratch"]

# Baseline method and the corresponding avialable dimensional settings
EXT_EVAL_FRAME = {
    "retfound": ["3D", "2D"],
    "from_scratch": ["3D", "2D"],
    "MAE-joint": ["3D"],
} 

# Baseline method and the corresponding default name for the output folder
EXPR_DEFAULT_NAME_DICT = {
    "retfound": ["outputs_ft", ""],
    "from_scratch": ["outputs_ft", "no_retFound"],
    "MAE-joint": ["outputs_ft_st", ""],
}

# Naming mapping of frame setting
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}

# Naming mapping of evaluation setting
SETTING_DICT = {"fewshot": "fewshot_10folds", "default": ""}

# Exteneded baseline methods with dimensionality and the plotting method name
# PLOT_METHODS_NAME = {
#     "MAE-joint 3D": "Ours model (3D MAE)",
#     "retfound 3D": "RETFound 3D",
#     "from_scratch 3D": "From scratch 3D",
#     "retfound 2D": "RETFound 2D",
#     "from_scratch 2D": "From scratch 2D",
# }

# -----------------END OF BASELINE SETTINGS-----------------


# -----------------MISC SUFFIX SETTINGS-----------------
# Miscellaneous suffix dictionary for the output folder
MISC_SUFFIX_DICT = {
    ("glaucoma", "fewshot", "outputs_ft", "2D"): "new_correct_visit",
    ("glaucoma", "fewshot", "outputs_ft", "3D"): "correct_visit",
    ("glaucoma", "default", "outputs_ft", "2D"): "new_correct_visit",
    ("glaucoma", "default", "outputs_ft", "3D"): "correct_visit",
    ("glaucoma", "fewshot", "outputs_ft_st", "3D"): "correct_visit",
    ("glaucoma", "default", "outputs_ft_st", "3D"): "correct_visit",
    ("duke14", "fewshot", "outputs_ft", "2D"): "effective_fold",
    ("duke14", "fewshot", "outputs_ft", "3D"): "effective_fold",
    ("duke14", "fewshot", "outputs_ft_st", "3D"): "effective_fold",
    ("hcms", "fewshot", "outputs_ft", "3D"): "correct_18",
    ("hcms", "default", "outputs_ft", "3D"): "correct_18",
    ("hcms", "fewshot", "outputs_ft", "2D"): "correct",
    ("hcms", "default", "outputs_ft", "2D"): "correct",
    ("hcms", "default", "outputs_ft_st", "3D"): "correct_18",
    ("hcms", "fewshot", "outputs_ft_st", "3D"): "correct_18",
    ("oimhs", "fewshot", "outputs_ft", "3D"): "correct_15",
    ("oimhs", "default", "outputs_ft", "3D"): "correct_15",
    ("oimhs", "fewshot", "outputs_ft", "2D"): "correct",
    ("oimhs", "default", "outputs_ft", "2D"): "correct",
    ("oimhs", "fewshot", "outputs_ft_st", "3D"): "correct_15",
    ("oimhs", "default", "outputs_ft_st", "3D"): "correct_15",
}

# Miscellaneous frame suffix dictionary for the output folder
MISC_FRAME_SUFFIX_DICT = {
    ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
}

# -----------------END OF MISC SUFFIX SETTINGS-----------------


# ----------CUSTOMIZED COLOR SETTINGS-----------------
boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}

# ---------- END OF CUSTOMIZED COLOR SETTINGS-----------------


def load_fold_results(file_path):
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
    return results

def get_results_json(out_dir, prefix='finetune_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"

def get_task_and_setting_grouped_dict(results_dict):
    setting_list = SETTING_DICT.keys()
    task_list = EXT_OPH_DATASET_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = EXT_OPH_DATASET_DICT[task]
            available_setting = EXT_OPH_DATASET_EVAL_SETTING[task]
            if setting not in available_setting:
                continue
            grouped_dict[setting][task] = {}
            for baseline in BASELINE:
                baseline_code = EXPR_DEFAULT_NAME_DICT[baseline][0]
                for frame in FRAME_DICT.keys():
                    # if (task_code, setting_val, baseline_code, frame) in results_dict:
                    #     grouped_dict[setting][task][(baseline, frame)] = results_dict[(task_code, setting_val, baseline, frame)]
                    if (task, setting, baseline, frame) in results_dict:
                        grouped_dict[setting][task][(baseline, frame)] = results_dict[(task, setting, baseline, frame)]

    for setting in setting_list:
        for task in task_list:
            if task not in grouped_dict[setting]:
                print(f"Missing {task} in {setting}")
    return grouped_dict




def ext_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1]):
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
    agg_ours = []
    agg_r3d = []
    y_max = 0
    y_max_line = 0
    xticks_list = []
    xticks_label = []
    for i, plot_task in enumerate(plot_tasks):
        ax = axes
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            y = np.mean(df_dict[plot_task][m][:, plot_col_idx])
            y_std_err = np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist()))
            handle = ax.bar(i * width * (len(plot_methods) + 1) + (j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name_key[j]], zorder=3)
            ax.errorbar(i * width * (len(plot_methods) + 1) + (j + 1) * width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            if y > best_y and j != 0:
                best_y = y
                compare_col = m
            if i == 0:  # Collect handle for legend only once per method across all tasks
                all_handles.append(handle)
                all_labels.append(plot_methods_name[j])
        agg_ours.append(np.mean(df_dict[plot_task][plot_methods[0]][:, plot_col_idx]))
        agg_r3d.append(np.mean(df_dict[plot_task][plot_methods[2]][:, plot_col_idx]))
        y_min = np.min([np.mean(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.5])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.4]) 
        
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max_temp = np.max([np.mean(df_dict[plot_task][m][:, plot_col_idx]) + np.std(df_dict[plot_task][m][:, plot_col_idx].tolist()) / \
                        np.sqrt(len(df_dict[plot_task][m][:, plot_col_idx].tolist())) for m in plot_methods])
        y_max = np.max([y_max, 0.98]) if setting_code == 'fewshot' else np.max([y_max, 1])
        print('y_max:', y_max, '\n\n\n' )
        y_h = df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        
        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        t_stat, p_value = ttest_rel(y_h * 2 , y_l * 2)
        # print(compare_col, plot_methods_name[0], p_value)

        xticks_list.append(i * width * (len(plot_methods) + 1) + width * (len(plot_methods) + 1) / 2)
        xticks_label.append(plot_task)
        ax.tick_params(axis='both', which='major', labelsize=18)
        # ax.set_xlabel(plot_task, fontsize=12)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=20)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = i * width * (len(plot_methods) + 1) + width
        x2 = i * width * (len(plot_methods) + 1) + (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=18, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)
        # ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        max_width = i * width * (len(plot_methods) + 1) + (j + 1) * width
    print('max_width:', max_width)
    # exit()
    ax.set_ylim(floor_to_nearest(y_min, 0.004), y_max)
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label, fontsize=20)
    ax.set_xlim(0.05, max_width + width)
    avg_ours = np.mean(agg_ours)
    avg_r3d = np.mean(agg_r3d)
    avg_improvement = avg_ours - avg_r3d
    avg_rel_improvement = avg_improvement / avg_r3d
    print(f'{plot_col}, Average improvement: {avg_improvement}, Average relative improvement: {avg_rel_improvement}', 'avg_ours:', avg_ours, 'avg_r3d:', avg_r3d)    
    return all_handles, all_labels
    # add legend for the axes
    


if __name__ == '__main__':
    results_dict = {}
    for dataset, dataset_code in EXT_OPH_DATASET_DICT.items():
        setting_list = EXT_OPH_DATASET_EVAL_SETTING[dataset] 
        for setting in setting_list:
            setting_val = SETTING_DICT[setting]
            for baseline in BASELINE:
                frame_list = EXT_EVAL_FRAME[baseline]
                frame_value = [FRAME_DICT[frame] for frame in frame_list]
                name_dict = EXPR_DEFAULT_NAME_DICT[baseline]
                out_folder = name_dict[0]
                expr = name_dict[1]
                for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                    frame = frame_list[i] # 3D, 2D
                    # print(name_dict)

                    suffix = ""
                    # print(out_folder)
                    if (dataset_code, setting, out_folder, frame) in MISC_SUFFIX_DICT:
                        suffix = MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                        # print(dataset, setting, out_folder, frame, suffix)
                    for fname in ["fold_results_test"]:
                        for ext in ["txt"]:
                            if (dataset_code, setting, out_folder) in MISC_FRAME_SUFFIX_DICT:
                                frame_val = MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                            file_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)
                            # print(file_path)
                            print(f"Loading {file_path}")
                            try:
                                result = load_fold_results(file_path)
                                print(result)
                                results_dict[(dataset, setting, baseline, frame)] = result
                            except:
                                print(f"Error loading {file_path}")
                                continue 
        print("\n")

    print("Results dict:", results_dict)
    grouped_dict = get_task_and_setting_grouped_dict(results_dict)


    # Plot the figure
    fig, axes = plt.subplots(figsize=(1.7*FIG_WIDTH, 1*FIG_HEIGHT), nrows=2, ncols=1)

    # plot the subfigure a-e
    ext_oph_tasks_barplot(fig, axes[0], grouped_dict, setting_code='fewshot', plot_col='auprc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUPRC') # auprc, Average improvement: 0.11126311360000019, Average relative improvement: 0.16845254565233772 avg_ours: 0.7717643436 avg_r3d: 0.6605012299999998
    import time 
    # time.sleep(10)
    # plot the subfigure f-j
    all_handles, all_labels = ext_oph_tasks_barplot(fig, axes[1], grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC') # auroc, Average improvement: 0.11160484419999994, Average relative improvement: 0.15939239997895133 avg_ours: 0.8117940892 avg_r3d: 0.700189245
    # fig.
    # plot the subfigure k
    fig.tight_layout( rect=[0, 0, 1, 0.97])
    # fig.tight_layout()
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=5, fontsize=13, frameon=False, columnspacing=0.8)


    
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2a-k_allinonebar.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2a-k_allinonebar.png'))
    import time 
    # time.sleep(10)
    fig, ax = plt.subplots(figsize=(1.7*FIG_WIDTH, 1*FIG_HEIGHT), nrows=2, ncols=1)
    ext_oph_tasks_barplot(fig, ax[0], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC') # auprc, Average improvement: 0.10738472466666671, Average relative improvement: 0.13508076692873475 avg_ours: 0.902351522 avg_r3d: 0.7949667973333333
    import time 
    # time.sleep(10)
    ext_oph_tasks_barplot(fig, ax[1], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC') # auroc, Average improvement: 0.11786051133333364, Average relative improvement: 0.14338705544766558 avg_ours: 0.9398350680000002 avg_r3d: 0.8219745566666665
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=5, fontsize=12, frameon=False, columnspacing=0.8)

    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l_allinonebar.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l_allinonebar.png'))

    # fig.tight_layout()
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l.pdf'), dpi=300)