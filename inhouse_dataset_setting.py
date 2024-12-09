import os
import numpy as np
import pandas as pd

home_dir = os.path.expanduser("~")
if 'wxdeng' in home_dir:
    this_file_dir = '/home/wxdeng/oph/retfound_baseline/'
else:
    this_file_dir = '/home/zucksliu/retfound_baseline/'

FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}

INHOUSE_SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}

# ----------CUSTOMIZED COLOR SETTINGS-----------------
# Exteneded baseline methods with dimensionality and the plotting method name
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    "from_scratch 3D": "Supervised (3D, w/o pre-training)",
    "retfound 2D": "RETFound (2D)",
    "from_scratch 2D": "Supervised (2D, w/o pre-training)",
    "imagenet 3D": "ImageNet trasferred 3D MAE",
}

boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}
# ---------- END OF CUSTOMIZED COLOR SETTINGS-----------------


# -----------------DATASET SETTINGS-----------------
INHOUSE_OPH_DATASET_DICT = {
    # "DUKE13": "duke13",
    # "MULTI_LABEL": "multi_label",
    "POG": "BCLS_POG",
    "DME": "BCLS_DME",
    "AMD": "BCLS_AMD",
    "ODR": "BCLS_ODR",
    "PM": "BCLS_PM",
    "CRO": "BCLS_CRO",
    "VD": "BCLS_VD",
    "RN": "BCLS_RN",
    }

INHOUSE_OPH_DATASET_EVAL_SETTING = dict([(key, ["fewshot", "default"]) for key in INHOUSE_OPH_DATASET_DICT.keys()])
print("Inhouse dataset eval setting: ", INHOUSE_OPH_DATASET_EVAL_SETTING)

# -----------------END OF DATASET SETTINGS-----------------


# -----------------BASELINE SETTINGS-----------------
INHOUSE_BASELINE = ["MAE-joint", "retfound", "imagenet", "from_scratch",]


# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "imagenet": ["3D"],
    "retfound": ["3D", "2D"],
    "from_scratch": ["3D", "2D"],
} 

INHOUSE_EXPR_DEFAULT_NAME_DICT = {
    "retfound 2D": ["outputs_ft_0529_ckpt_flash_attn", "retfound"],
    "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
    "retfound 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_retfound"],
    "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
    "MAE-joint 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_3d_256_0509"],
    "imagenet 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_3d_imagenet"],
}
INHOUSE_MISC_SUFFIX_DICT = {}
INHOUSE_MISC_FRAME_SUFFIX_DICT = {}





def inhouse_load_fold_results(file_path):
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
                numbers = line.strip().split('[')[2].split(']')[0]  # Extract numbers between brackets

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


def inhouse_get_results_json(out_dir, prefix='finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"



def get_inhouse_task_and_setting_grouped_dict(results_dict, results_csv_dict):
    setting_list = INHOUSE_SETTING_DICT.keys()
    task_list = INHOUSE_OPH_DATASET_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = INHOUSE_SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = INHOUSE_OPH_DATASET_DICT[task]
            available_setting = INHOUSE_OPH_DATASET_EVAL_SETTING[task]
            if setting not in available_setting:
                continue
            grouped_dict[setting][task] = {}
            for baseline in INHOUSE_BASELINE:

                for frame in FRAME_DICT.keys():
                    baseline_plus_frame = f"{baseline} {frame}"

                    if (task, setting, baseline, frame) in results_dict:
                        grouped_dict[setting][task][(baseline, frame)] = [results_dict[(task, setting, baseline, frame)], results_csv_dict[(task, setting, baseline, frame)]]
    # print(grouped_dict)
    for setting in setting_list:
        for task in task_list:
            if task not in grouped_dict[setting]:
                print(f"Missing {task} in {setting}")
    return grouped_dict



def find_candidate_metric(metric, metric_csv_dict):
    candidate_metric_dict = {'auprc':['AUPRC', 'auprc', 'aucpr', 'auc_pr'], 'auroc':['AUROC', 'auroc', 'ROC AUC', 'auc_roc'], 'acc':['Accuracy', 'acc'], 'bal_acc':['Balanced Accuracy', 'bal_acc', 'Balanced Acc']}
    column_names = metric_csv_dict.columns
    candidate_list = candidate_metric_dict[metric]
    for candidate in candidate_list:
        if candidate in column_names:
            return candidate
    return None


inhouse_results_dict = {}
inhouse_results_csv_dict = {}
for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
    setting_list = INHOUSE_OPH_DATASET_EVAL_SETTING[dataset] 
    for setting in setting_list:
        setting_val = INHOUSE_SETTING_DICT[setting]
        for baseline in INHOUSE_BASELINE:
            frame_list = INHOUSE_EVAL_FRAME[baseline]
            frame_value = [FRAME_DICT[frame] for frame in frame_list]

            for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                frame = frame_list[i] # 3D, 2D
                # print(name_dict)
                baseline_plus_frame = f"{baseline} {frame}"
                name_dict = INHOUSE_EXPR_DEFAULT_NAME_DICT[baseline_plus_frame]
                out_folder = name_dict[0]
                expr = name_dict[1]
                suffix = ""
                # print(out_folder)
                if (dataset_code, setting, out_folder, frame) in INHOUSE_MISC_SUFFIX_DICT:
                    suffix = INHOUSE_MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                    # print(dataset, setting, out_folder, frame, suffix)
                
                # for fname in ["metrics_test_singlefold"]:
                #     # for ext in ["csv"]:
                fname = 'metrics_test_singlefold'
                replace_fname = 'macro_metrics_test_singlefold'
                if (dataset_code, setting, out_folder) in INHOUSE_MISC_FRAME_SUFFIX_DICT:
                    frame_val = INHOUSE_MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                file_path = inhouse_get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext='csv')
                try:
                    result = pd.read_csv(file_path) 
                    inhouse_results_csv_dict[(dataset, setting, baseline, frame)] = result
                except:
                    print(f"Error loading {file_path}")
                    replace_path = inhouse_get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='csv')
                    print(f"Loading {replace_path}")
                    result = pd.read_csv(replace_path) # result = load_fold_results(replace_path)
                    print(result)
                    inhouse_results_csv_dict[(dataset, setting, baseline, frame)] = result
                # print(f"Loading {file_path}")
                fname = 'fold_results_test'
                replace_fname = 'fold_results_test_for_best_val'
                file_path = inhouse_get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext='txt')
                try:
                    result = inhouse_load_fold_results(file_path)
                    print(result)
                    inhouse_results_dict[(dataset, setting, baseline, frame)] = result
                except:
                    print(f"Error loading {file_path}")
                    replace_path = inhouse_get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='txt')
                    print(f"Loading {replace_path}")
                    result = pd.read_csv(replace_path) # result = load_fold_results(replace_path)
                    print(result)
                    inhouse_results_dict[(dataset, setting, baseline, frame)] = result
                    continue 
    print("\n")
print(inhouse_results_csv_dict)

print(inhouse_results_dict)
inhouse_grouped_dict = get_inhouse_task_and_setting_grouped_dict(inhouse_results_dict, inhouse_results_csv_dict)
print(inhouse_grouped_dict)