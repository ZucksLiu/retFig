import os
import numpy as np
import pandas as pd
import json

this_file_dir = '/home/zucksliu/retfound_baseline/'
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
save_file_dir = os.path.dirname(os.path.abspath(__file__))

PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    # "from_scratch 3D": "From scratch 3D",
    "retfound 2D": "RETFound (2D)",
    # "from_scratch 2D": "From scratch 2D",
}

INHOUSE_NON_OPH_DATASET_DICT = {
    "MULTI_LABEL": "multi_label",
}

INHOUSE_NON_OPH_MISC_SUFFIX_DICT = {}
INHOUSE_NON_OPH_MISC_FRAME_SUFFIX_DICT = {}

INHOUSE_NON_OPH_DATASET_EVAL_SETTING = dict([(key, ["default"]) for key in INHOUSE_NON_OPH_DATASET_DICT.keys()])

INHOUSE_NON_OPH_BASELINE = ["MAE-joint", "retfound"]#, "from_scratch"]

INHOUSE_NON_OPH_SETTING_DICT = {"default": "correct_patient"} # "fewshot": "correct_patient_fewshot", 

INHOUSE_NON_OPH_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "retfound": ["3D", "2D"],
    # "from_scratch": ["3D", "2D"],
} 

INHOUSE_NON_OPH_RUNS_DICT = {
    1: "runs_1",
    2: "runs_2",
    3: "runs_3",
    4: "runs_4",
    5: "runs_5",
}


INHOUSE_NON_OPH_EXPR_DEFAULT_NAME_DICT = {
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


def inhouse_non_oph_get_results_json(out_dir, runs, prefix='l1_100_finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + runs + '/' + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"


def inhouse_non_oph_load_fold_results(file_path):
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


def get_inhouse_non_oph_run_max_val_col(macro_results_dict, metric='AUPRC'):
    max_val_dict = {}
    for key, result in macro_results_dict.items():
        print(result)
        print(key)

        col = result[metric].tolist()
        print(col, len(col), np.argmax(col))
        max_val_dict[key] = np.argmax(col)
        # break
    return max_val_dict

def get_inhouse_non_oph_task_and_setting_grouped_dict(results_dict):
    setting_list = INHOUSE_NON_OPH_SETTING_DICT.keys()
    grouped_dict = {}
    for dataset, dataset_code in INHOUSE_NON_OPH_DATASET_DICT.items():
        for setting in setting_list:
            setting_val = INHOUSE_NON_OPH_SETTING_DICT[setting]
            grouped_dict[setting] = {}
            for task in TASKS.keys():
                available_setting = INHOUSE_NON_OPH_DATASET_EVAL_SETTING[dataset]
                print('Available setting:', available_setting)
                if setting not in available_setting:
                    continue
                grouped_dict[setting][task] = {}
                for baseline in INHOUSE_NON_OPH_BASELINE:

                    for frame in FRAME_DICT.keys():
                        baseline_plus_frame = f"{baseline} {frame}"
                        if (dataset, setting, 'runs_1', task, baseline, frame) in results_dict:
                            grouped_dict[setting][task][(baseline, frame)] = dict()
                        else:
                            continue
                        for runs in INHOUSE_NON_OPH_RUNS_DICT.values():

                            grouped_dict[setting][task][(baseline, frame)][runs] = results_dict[(dataset, setting, runs, task, baseline, frame)]
                            # grouped_dict[setting][task][(baseline, frame)] = results_dict[(task, setting, baseline, frame)]

    return grouped_dict


def get_inhouse_non_oph_organized_run_results(grouped_dict, run_max_val_dict, mode='test'):
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
                for runs in INHOUSE_NON_OPH_RUNS_DICT.values():
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

    print(organized_dict)
    return organized_dict



inhouse_non_oph_results_dict = {}
inhouse_non_oph_macro_results_dict = {}
print(INHOUSE_NON_OPH_DATASET_DICT)
print(INHOUSE_NON_OPH_DATASET_EVAL_SETTING)
print(INHOUSE_NON_OPH_BASELINE)
# exit()
for dataset, dataset_code in INHOUSE_NON_OPH_DATASET_DICT.items():
    setting_list = INHOUSE_NON_OPH_DATASET_EVAL_SETTING[dataset] 
    for setting in setting_list:
        setting_val = INHOUSE_NON_OPH_SETTING_DICT[setting]
        for baseline in INHOUSE_NON_OPH_BASELINE:
            frame_list = INHOUSE_NON_OPH_EVAL_FRAME[baseline]
            frame_value = [FRAME_DICT[frame] for frame in frame_list]
            print(f"Dataset: {dataset}, Setting: {setting}, Baseline: {baseline}, Frames: {frame_list}", frame_value)
            for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                frame = frame_list[i] # 3D, 2D
                # print(name_dict)
                baseline_plus_frame = f"{baseline} {frame}"
                name_dict = INHOUSE_NON_OPH_EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
                out_folder = name_dict[0]
                expr = name_dict[1]
                suffix = ""
                # print(out_folder)
                if (dataset_code, setting, out_folder, frame) in INHOUSE_NON_OPH_MISC_SUFFIX_DICT:
                    suffix = INHOUSE_NON_OPH_MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                    # print(dataset, setting, out_folder, frame, suffix)
                for runs in INHOUSE_NON_OPH_RUNS_DICT.values():
                    macro_fname = f'macro_metrics_val_singlefold'
                    macro_file_path = inhouse_non_oph_get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=macro_fname, ext='csv')
                    macro_result = pd.read_csv(macro_file_path)
                    inhouse_non_oph_macro_results_dict[(dataset, setting, runs, baseline, frame)] = macro_result
                    for task in TASKS:
                        task_idx = TASKS_IDX[task]
                        fname = f'class_{task_idx}_{task}_metrics_test_singlefold'
                        replace_fname = 'fold_results_test'
                        # for fname in ["fold_results_test"]:
                        for ext in ["csv"]:
                            if (dataset_code, setting, out_folder) in INHOUSE_NON_OPH_MISC_FRAME_SUFFIX_DICT:
                                frame_val = INHOUSE_NON_OPH_MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                            file_path = inhouse_non_oph_get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)

                            # print(f"Loading {file_path}")
                            try:
                                # result = load_fold_results(file_path)
                                result = pd.read_csv(file_path)
                                print('res:', result)
                                inhouse_non_oph_results_dict[(dataset, setting, runs, task, baseline, frame)] = result
                            except:
                                print(f"Error loading {file_path}")
                                replace_path = inhouse_non_oph_get_results_json(this_file_dir + out_folder + '/', runs=runs, dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='txt')
                                print(f"Loading {replace_path}")
                                result = inhouse_non_oph_load_fold_results(replace_path)
                                print('replace res:', result)
                                inhouse_non_oph_results_dict[(dataset, setting, runs, task, baseline, frame)] = result
                                continue 
    print("\n")
# exit()
print(inhouse_non_oph_results_dict)
inhouse_non_oph_grouped_dict = get_inhouse_non_oph_task_and_setting_grouped_dict(inhouse_non_oph_results_dict)
print(inhouse_non_oph_grouped_dict)
inhouse_non_oph_max_val_dict = get_inhouse_non_oph_run_max_val_col(inhouse_non_oph_macro_results_dict)
inhouse_non_oph_organized_dict = get_inhouse_non_oph_organized_run_results(inhouse_non_oph_grouped_dict, inhouse_non_oph_max_val_dict)

inhouse_non_oph_organized_dict_joint_name = {}
for task, task_dict in inhouse_non_oph_organized_dict['default'].items():
    inhouse_non_oph_organized_dict_joint_name[task] = {}
    for baseline_and_frame, runs_dict in task_dict.items():
        baseline, frame = baseline_and_frame
        baseline_plus_frame = f"{baseline} {frame}"
        inhouse_non_oph_organized_dict_joint_name[task][baseline_plus_frame] = runs_dict

print('grouped_dict:', inhouse_non_oph_grouped_dict['default']['E11'])
print('organized_dict:', inhouse_non_oph_organized_dict)
save_path = os.path.join(save_file_dir, 'save_figs', 'inhouse_non_oph_organized_dict_joint_name.json')
with open(save_path, 'w') as f:
    json.dump(inhouse_non_oph_organized_dict_joint_name, f, indent=2)
    