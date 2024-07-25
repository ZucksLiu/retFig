import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind
import time

this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))

# # -----------------DATASET SETTINGS-----------------
INHOUSE_OPH_DATASET_DICT = {
    "MULTI_LABEL": "multi_label",
#     "POG": "BCLS_POG",
#     "DME": "BCLS_DME",
#     "AMD": "BCLS_AMD",
#     "ODR": "BCLS_ODR",
#     "PM": "BCLS_PM",
#     "CRO": "BCLS_CRO",
#     "VD": "BCLS_VD",
#     "RN": "BCLS_RN",
}
MISC_SUFFIX_DICT = {}
MISC_FRAME_SUFFIX_DICT = {}

INHOUSE_OPH_DATASET_EVAL_SETTING = dict([(key, ["default"]) for key in INHOUSE_OPH_DATASET_DICT.keys()])
# print("Inhouse dataset eval setting: ", INHOUSE_OPH_DATASET_EVAL_SETTING)
# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
BASELINE = ["MAE-joint", "retfound"]#, "from_scratch"]
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
SETTING_DICT = {"default": "correct_patient"} # "fewshot": "correct_patient_fewshot", 

# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "retfound": ["3D", "2D"],
    # "from_scratch": ["3D", "2D"],

} 

RUNS_DICT = {
    1: "runs_1",
    2: "runs_2",
    3: "runs_3",
    4: "runs_4",
    5: "runs_5",
}

EXPR_DEFAULT_NAME_DICT = {
    "retfound 2D": ["non_oph_outputs_ft_2d_0621_more_runs_ckpt_flash_attn", "singlefold_retfound"],
    # "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
    "retfound 3D": ["non_oph_outputs_ft_st_0621_more_runs_ckpt_flash_attn", "singlefold_retfound"],
    # "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
    "MAE-joint 3D": ["non_oph_outputs_ft_st_0621_more_runs_ckpt_flash_attn", "singlefold_3d_256_0509"],
}




TASKS = {
    "E11":"Diabetes",
    "I10":"Hypertension",
    "M25":"Joint pain",
    "E78":"Hyperlipidemia",
    "M79":"Soft tissue\n disorders",
    "M54":"Back pain",
    "G89":"Pain",
}

TASKS_IDX = {
    "E11":9,
    "I10":3,
    "M25":0,
    "E78":8,
    "M79":4,
    "M54":6,
    "G89":11,
}

# Exteneded baseline methods with dimensionality and the plotting method name
# PLOT_METHODS_NAME = {
#     "MAE-joint": "Ours model (3D MAE)",
#     "retFound 3D": "RETFound 3D",
#     # "from_scratch 3D": "From scratch 3D",
#     "retFound 2D": "RETFound 2D",
#     # "from_scratch 2D": "From scratch 2D",
# }
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    # "from_scratch 3D": "From scratch 3D",
    "retfound 2D": "RETFound (2D)",
    # "from_scratch 2D": "From scratch 2D",
}

def get_results_json(out_dir, runs, prefix='l1_100_finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + runs + '/' + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"

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

def get_inhouse_run_max_val_col(macro_results_dict, metric='AUPRC'):
    max_val_dict = {}
    for key, result in macro_results_dict.items():
        print(result)
        print(key)

        col = result[metric].tolist()
        print(col, len(col), np.argmax(col))
        max_val_dict[key] = np.argmax(col)
        # break
    return max_val_dict

def get_inhouse_task_and_setting_grouped_dict(results_dict):
    setting_list = SETTING_DICT.keys()
    # task_list = INHOUSE_OPH_DATASET_DICT.keys()
    grouped_dict = {}
    for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
        for setting in setting_list:
            setting_val = SETTING_DICT[setting]
            grouped_dict[setting] = {}
            for task in TASKS.keys():
                # task_code = INHOUSE_OPH_DATASET_DICT[task]
                available_setting = INHOUSE_OPH_DATASET_EVAL_SETTING[dataset]
                print('Available setting:', available_setting)
                if setting not in available_setting:
                    continue
                grouped_dict[setting][task] = {}
                for baseline in BASELINE:

                    for frame in FRAME_DICT.keys():
                        baseline_plus_frame = f"{baseline} {frame}"
                        if (dataset, setting, 'runs_1', task, baseline, frame) in results_dict:
                            grouped_dict[setting][task][(baseline, frame)] = dict()
                        else:
                            continue
                        for runs in RUNS_DICT.values():

                        # baseline_code = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame][0]
                        # if (task_code, setting_val, baseline_code, frame) in results_dict:
                        #     grouped_dict[setting][task][(baseline, frame)] = results_dict[(task_code, setting_val, baseline, frame)]
                                # ataset, setting, runs, task, baseline, frame

                            grouped_dict[setting][task][(baseline, frame)][runs] = results_dict[(dataset, setting, runs, task, baseline, frame)]
                            # grouped_dict[setting][task][(baseline, frame)] = results_dict[(task, setting, baseline, frame)]
    # print(grouped_dict)
    # for setting in setting_list:
    #     for task in task_list:
    #         if task not in grouped_dict[setting]:
    #             print(f"Missing {task} in {setting}")
    return grouped_dict


def get_inhouse_organized_run_results(grouped_dict, run_max_val_dict, mode='test'):
    selected_metrics = ['ROC AUC', 'AUPRC', 'Accuracy', 'Balanced Acc']
    organized_dict = {}
    for setting, setting_dict in grouped_dict.items():
        organized_dict[setting] = {}
        for task, task_dict in setting_dict.items():
            organized_dict[setting][task] = {}
            for baseline_and_frame, runs_dict in task_dict.items():
                # for frame, runs_dict in frame_dict.items():
                baseline, frame = baseline_and_frame
                baseline_plus_frame = f"{baseline} {frame}"
                    # if (baseline, frame) not in organized_dict[setting][task]:
                    #     continue
                    
                    
                organized_dict[setting][task][(baseline, frame)] = []
                for runs in RUNS_DICT.values():
                    if runs not in grouped_dict[setting][task][(baseline, frame)]:
                        continue
                    if mode == 'test':
                        test_results = grouped_dict[setting][task][(baseline, frame)][runs]
                        max_val_col = run_max_val_dict[(dataset, setting, runs, baseline, frame)]
                        test_results_column = test_results.columns
                        test_results = test_results.iloc[max_val_col * 2]
                        test_results = test_results.to_dict()
                        test_results = {k: float(test_results[k]) for k in selected_metrics}
                        test_results = list(test_results.values())
                        organized_dict[setting][task][(baseline, frame)].append(test_results)
                        print(test_results)
                # make the organized_dict[setting][task][(baseline, frame)] to be a dataframe
                # organized_dict[setting][task][(baseline, frame)] = pd.DataFrame(organized_dict[setting][task][(baseline, frame)], columns=selected_metrics)


    # for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
    #     for setting in SETTING_DICT.keys():
    #         for task in TASKS.keys():
    #             for baseline in BASELINE:
    #                 for frame in FRAME_DICT.keys():
    #                     baseline_plus_frame = f"{baseline} {frame}"
    #                     if (baseline, frame) not in grouped_dict[setting][task]:
    #                         continue
    #                     organized_dict[setting] = {}
    #                     organized_dict[setting][task] = {}
    #                     organized_dict[setting][task][(baseline, frame)] = []
    #                     for runs in RUNS_DICT.values():
    #                         if runs not in grouped_dict[setting][task][(baseline, frame)]:
    #                             continue
    #                         if mode == 'test':
    #                             test_results = grouped_dict[setting][task][(baseline, frame)][runs]
    #                             max_val_col = run_max_val_dict[(dataset, setting, runs, baseline, frame)]
    #                             test_results_column = test_results.columns
    #                             test_results = test_results.iloc[max_val_col * 2]
    #                             test_results = test_results.to_dict()
    #                             test_results = {k: float(test_results[k]) for k in selected_metrics}
    #                             test_results = list(test_results.values())
    #                             organized_dict[setting][task][(baseline, frame)].append(test_results)
    #                             print(test_results)
    print(organized_dict)
    return organized_dict

def INHOUSE_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
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
    # print(plot_tasks, plot_methods)
    # exit()

    if plot_methods_name is None:
        plot_methods_name_key = [m[0] + ' ' + m[1] for m in plot_methods]
        plot_methods_name = [PLOT_METHODS_NAME[m] for m in plot_methods_name_key]
        print(plot_methods_name)
    else:
        # TODO: less flexible as ultimately the plot_methods_name_key need also to be provided
        plot_methods_name_key = plot_methods

    plot_col_dict = ['auroc', 'auprc', 'acc', 'bal_acc']
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
        cur_top5_val = []
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            result = df_dict[plot_task][m]
            print(result, plot_col_idx, type(plot_col_idx))
            result = np.array(result)
            y = np.mean(result[:, plot_col_idx])
            # print(result_csv.columns)
            # plot_col_csv = find_candidate_metric(plot_col, result_csv)
            # if plot_col_csv is None:
            #     print(f"Cannot find {plot_col} in {result_csv.columns}")
            #     exit()
            
            # y_other_val = result_csv[plot_col_csv]
            # for idx, val in enumerate(y_other_val):
            #     try:
            #         y_other_val[idx] = float(val)
            #     except:
            #         y_other_val[idx] = 0.0
            # try:
            #     y_other_val = y_other_val.astype(float)
            # except:
            #     print('Exist error in converting y_other_val to float')
            #     exit()
            # print('y_other_val before error', y_other_val, type(y_other_val))
            # y_other_val_top5 = list(y_other_val.nlargest(5))
            # print(y, 'y_other_val', y_other_val, y_other_val_top5, type(y_other_val_top5)) 
            # cur_top5_val.append(y_other_val_top5)
            # exit()
            handle = ax.bar(i * width * (len(plot_methods) + 1) + (j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name_key[j]], zorder=3)
            if err_bar:
                y_std_err = np.std(result[:, plot_col_idx]) / \
                        np.sqrt(len(result))
                ax.errorbar(i * width * (len(plot_methods) + 1) + (j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)

            if y > best_y and j != 0:
                best_y = y
                compare_col = m
                compare_col_idx = j
            if i == 0:  # Collect handle for legend only once per method across all tasks
                all_handles.append(handle)
                all_labels.append(plot_methods_name[j])
        xticks_list.append(i * width * (len(plot_methods) + 1) + width * (len(plot_methods) + 1) / 2)
        xticks_label.append(TASKS[plot_task])
        ax.tick_params(axis='x', which='major', labelsize=15)
        agg_ours.append(np.mean(np.array(df_dict[plot_task][plot_methods[0]])[:, plot_col_idx]))
        agg_r3d.append(np.mean(np.array(df_dict[plot_task][plot_methods[1]])[:, plot_col_idx]))
        print('agg_ours:', agg_ours, 'agg_r3d:', agg_r3d)
        print(np.array(df_dict[plot_task][m][0]))
        y_min = np.min([np.mean(np.array(df_dict[plot_task][m])[:, plot_col_idx]) for m in plot_methods])
        y_max = np.max([np.mean(np.array(df_dict[plot_task][m])[:, plot_col_idx]) + np.std(np.array(df_dict[plot_task][m])[:, plot_col_idx]) / np.sqrt(len(df_dict[plot_task][m])) for m_idx, m in enumerate(plot_methods)])

        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.5])
            y_max = np.max([y_max, 0.85])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.3]) 
            y_max = np.max([y_max, 0.65])
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])

        y_h = np.array(df_dict[plot_task][plot_methods[0]])[:, plot_col_idx].tolist()
        y_l = np.array(df_dict[plot_task][compare_col])[:, plot_col_idx].tolist()
        
        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        t_stat, p_value = ttest_rel(y_h , y_l )
        # print(compare_col, plot_methods_name[0], p_value)

        # ax.set_xticks([])
        # ax.set_xlabel(TASKS[plot_task], fontsize=15)

        if i == 0:
            ax.set_ylabel(y_name, fontsize=20)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', which='major', labelsize=18)
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
            ax.text((x1 + x2)/2, line_y, stars, fontsize=20, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)
        # ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        ax.set_ylim(y_min, y_max)
        max_width = i * width * (len(plot_methods) + 1) + (len(plot_methods) + 1) * width
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label, fontsize=20)
    print('max_width', max_width)
    ax.set_xlim(0.05, max_width)
    avg_ours = np.mean(agg_ours)
    avg_r3d = np.mean(agg_r3d)
    avg_improvement = avg_ours - avg_r3d
    avg_rel_improvement = avg_improvement / avg_r3d
    print(f'{plot_col}, Average improvement: {avg_improvement}, Average relative improvement: {avg_rel_improvement}', 'avg_ours:', avg_ours, 'avg_r3d:', avg_r3d)
    return all_handles, all_labels
    # add legend for the axes
    



def plot_icd10_expr_results_more_run(axes, results, plot_col='AUPRC', plot_name='AUPRC'):
    
    all_handles = []
    all_labels = []
    for i, task in enumerate(TASKS):
        ax = axes[i]
        df = results[task]
        y = df[plot_col].tolist()
        
        method = df[task].tolist()
        width = 0.8 / len(method)
        # print(method, y)
        for j, m in enumerate(method):
            handle = ax.bar((j + 1)*width, y[j], width, label=PLOT_METHODS_NAME[m], color=COLORS[m], zorder=3)
            if i == 0:
                all_handles.append(handle)
                all_labels.append(PLOT_METHODS_NAME[m])
        ax.set_xticks([])
        # ax.set_xlabel(task + ': ' + TASKS[task])
        if i == 0:
            ax.set_ylabel(plot_name)
        # y_max = np.max(y)
        # y_min = np.min(y)
        y_max = np.max(y) + 0.05

        if plot_col == 'AUPRC':
            ax.set_ylim(0.3, y_max)
        elif plot_col == 'ROC AUC':
            ax.set_ylim(0.55, y_max)
        else:
            ax.set_ylim(0.3, 0.7)
        format_ax(ax)
    return all_handles, all_labels

def plot_icd10_expr(results):
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=7)
    plot_icd10_expr_results(ax[0, :], results, plot_col='ROC AUC', plot_name='AUROC')
    all_handles, all_labels = plot_icd10_expr_results(ax[1, :], results, plot_col='AUPRC', plot_name='AUPRC')
    
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'icd10_expr_results.png'))
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'icd10_expr_results.pdf'), dpi=300)
    # Draw bar plot for each metric

if __name__ == '__main__':
    # results = {}
    # for task in TASKS:
    #     print(task)
    #     task_csv_dir = os.path.join(save_file_dir, f'ICD10-expr/{task}.csv')
    #     task_df = pd.read_csv(task_csv_dir, encoding='utf-8')
    #     results[task] = task_df
    # print('Results:', results)
    # plot_icd10_expr(results)
    results_dict = {}
    macro_results_dict = {}
    print(INHOUSE_OPH_DATASET_DICT)
    print(INHOUSE_OPH_DATASET_EVAL_SETTING)
    print(BASELINE)
    # exit()
    for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
        setting_list = INHOUSE_OPH_DATASET_EVAL_SETTING[dataset] 
        for setting in setting_list:
            setting_val = SETTING_DICT[setting]
            for baseline in BASELINE:
                frame_list = INHOUSE_EVAL_FRAME[baseline]
                frame_value = [FRAME_DICT[frame] for frame in frame_list]
                print(f"Dataset: {dataset}, Setting: {setting}, Baseline: {baseline}, Frames: {frame_list}", frame_value)
                for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                    frame = frame_list[i] # 3D, 2D
                    # print(name_dict)
                    baseline_plus_frame = f"{baseline} {frame}"
                    name_dict = EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
                    out_folder = name_dict[0]
                    expr = name_dict[1]
                    suffix = ""
                    # print(out_folder)
                    if (dataset_code, setting, out_folder, frame) in MISC_SUFFIX_DICT:
                        suffix = MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                        # print(dataset, setting, out_folder, frame, suffix)
                    for runs in RUNS_DICT.values():
                        macro_fname = f'macro_metrics_val_singlefold'
                        macro_file_path = get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=macro_fname, ext='csv')
                        macro_result = pd.read_csv(macro_file_path)
                        macro_results_dict[(dataset, setting, runs, baseline, frame)] = macro_result
                        for task in TASKS:
                            task_idx = TASKS_IDX[task]
                            fname = f'class_{task_idx}_{task}_metrics_test_singlefold'
                            replace_fname = 'fold_results_test'
                            # for fname in ["fold_results_test"]:
                            for ext in ["csv"]:
                                if (dataset_code, setting, out_folder) in MISC_FRAME_SUFFIX_DICT:
                                    frame_val = MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                                file_path = get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)

                                # print(f"Loading {file_path}")
                                try:
                                    # result = load_fold_results(file_path)
                                    result = pd.read_csv(file_path)
                                    print('res:', result)
                                    results_dict[(dataset, setting, runs, task, baseline, frame)] = result
                                except:
                                    print(f"Error loading {file_path}")
                                    replace_path = get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='txt')
                                    print(f"Loading {replace_path}")
                                    result = load_fold_results(replace_path)
                                    print('replace res:', result)
                                    results_dict[(dataset, setting, runs, task, baseline, frame)] = result
                                    continue 
        print("\n")
    # exit()
    print(results_dict)
    grouped_dict = get_inhouse_task_and_setting_grouped_dict(results_dict)
    print(grouped_dict)
    max_val_dict = get_inhouse_run_max_val_col(macro_results_dict)
    organized_dict = get_inhouse_organized_run_results(grouped_dict, max_val_dict)
    print('grouped_dict:', grouped_dict['default']['E11'])
    print('organized_dict:', organized_dict)
    # Plot the figure
    fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 1*FIG_HEIGHT), nrows=2, ncols=1)

    # results = {}
    # for task in TASKS:
        
    #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
    #     df_dict = {}
    #     for exp_code in EXP_CODE_DICT.keys():
    #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
    #     results[TASKS[task]] = df_dict

    # plot the subfigure a-e
    INHOUSE_oph_tasks_barplot(fig, axes[0], organized_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC')
    # time.sleep(10) # auprc, Average improvement: 0.019488472878051888, Average relative improvement: 0.040658527219800566 avg_ours: 0.49880915197439324 avg_r3d: 0.47932067909634135
    # plot the subfigure f-j
    all_handles, all_labels = INHOUSE_oph_tasks_barplot(fig, axes[1], organized_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC') # auroc, Average improvement: 0.022644281589650928, Average relative improvement: 0.03345673301663846 avg_ours: 0.6994671374969996 avg_r3d: 0.6768228559073487
    # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
    # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
    fig.tight_layout(rect=[0, 0.0, 1, 0.97])
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=3, fontsize=20, frameon=False)
    

    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'icd10_expr_more_runs_ci_allinonebar.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'icd10_expr_more_runs_ci_allinonebar.png'))    
    exit()



# def plot_retclip_recall_and_mean_rank(axes, retclip_exp_res_df, prefix, col_names, reverse_plot=True):
#     plot_method = retclip_exp_res_df['Method'].tolist()
#     print(plot_method)
#     full_col_names = [f'{prefix} {col_name}' for col_name in col_names]
#     print(full_col_names)
#     all_handles = []
#     all_labels = []
#     for i, col_name in enumerate(col_names):
#         ax = axes[i]
#         y = retclip_exp_res_df[full_col_names[i]].tolist()
#         print(y)
        
#         width = 0.8 / len(plot_method)
#         if reverse_plot:
#             y = y[::-1]
#         for j in range(len(plot_method)):
#             method = plot_method[::-1][j]
#             handle = ax.bar((j + 1)*width, y[j], width, label=method, color=COLORS[method], zorder=3,)
#             if i == 0:
#                 all_handles.append(handle)
#                 all_labels.append(method)
#                 print(all_labels, all_handles)
#         ax.set_xticks([])
#         ax.set_xlabel(col_name, fontsize=8)
#         if i == 0:
#             ax.set_ylabel(prefix.replace('_', ' '), fontsize=8)
#         y_max = np.max(y)
#         y_min = np.min(y)
#         print('y_max', y_max, 'y_min', y_min)
#         if col_name == 'mean rank':
#             ax.set_ylim(0, y_max + 5)
#         else:
#             ax.set_ylim(floor_to_nearest(y_min, 0.004) - 0.1, y_max + 0.1)
#         format_ax(ax)
#     return all_handles, all_labels

#     # # handle = ax.bar((j + 1)*width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name[j]], zorder=3)
#     # exit()
#     # return ax

# def plot_retclip_exp_res(retclip_exp_res_df):
#     fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
#     first_row_prefix = 'OCT_to_IR'
#     second_row_prefix = 'IR_to_OCT'
#     col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
#     all_handles, all_labels = plot_retclip_recall_and_mean_rank(ax[0, :], retclip_exp_res_df, first_row_prefix, col_names)
#     plot_retclip_recall_and_mean_rank(ax[1, :], retclip_exp_res_df, second_row_prefix, col_names)
#     fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8)
#     fig.tight_layout()
#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res.png'))
#     plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res.pdf'), dpi=300)
#     # Draw bar plot for each metric
    
# plot_retclip_exp_res(retclip_exp_res_df)

# exit()


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



# def get_results_json(out_dir, prefix='finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
#     if setting != "":
#         setting = f"_{setting}"
#     if expr != "":
#         expr = f"_{expr}"
#     if suffix != "":
#         expr = f"{expr}_{suffix}"
#         # print(expr, suffix)
#     return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"






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