import os
import numpy as np

home_dir = os.path.expanduser("~")
if 'wxdeng' in home_dir:
    this_file_dir = '/home/wxdeng/oph/retfound_baseline/'
else:
    this_file_dir = '/home/zucksliu/retfound_baseline/'
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
# Naming mapping of evaluation setting
EXT_OPH_SETTING_DICT = {"fewshot": "fewshot_10folds", "default": ""}

# ----------CUSTOMIZED COLOR SETTINGS-----------------
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OctCube",
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
EXT_OPH_BASELINE = ["MAE-joint", "retfound", "from_scratch"]

# Baseline method and the corresponding avialable dimensional settings
EXT_OPH_EVAL_FRAME = {
    "retfound": ["3D", "2D"],
    "from_scratch": ["3D", "2D"],
    "MAE-joint": ["3D"],
} 

# Baseline method and the corresponding default name for the output folder
EXT_OPH_EXPR_DEFAULT_NAME_DICT = {
    "retfound": ["outputs_ft", ""],
    "from_scratch": ["outputs_ft", "no_retFound"],
    "MAE-joint": ["outputs_ft_st", ""],
}



# -----------------MISC SUFFIX SETTINGS-----------------
# Miscellaneous suffix dictionary for the output folder
EXT_OPH_MISC_SUFFIX_DICT = {
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
EXT_OPH_MISC_FRAME_SUFFIX_DICT = {
    ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
}

# -----------------END OF MISC SUFFIX SETTINGS-----------------


def ext_oph_load_fold_results(file_path):
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



def ext_oph_get_results_json(out_dir, prefix='finetune_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"



def get_ext_oph_task_and_setting_grouped_dict(results_dict):
    setting_list = EXT_OPH_SETTING_DICT.keys()
    task_list = EXT_OPH_DATASET_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = EXT_OPH_SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = EXT_OPH_DATASET_DICT[task]
            available_setting = EXT_OPH_DATASET_EVAL_SETTING[task]
            if setting not in available_setting:
                continue
            grouped_dict[setting][task] = {}
            for baseline in EXT_OPH_BASELINE:
                baseline_code = EXT_OPH_EXPR_DEFAULT_NAME_DICT[baseline][0]
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



ext_oph_results_dict = {}
for dataset, dataset_code in EXT_OPH_DATASET_DICT.items():
    setting_list = EXT_OPH_DATASET_EVAL_SETTING[dataset] 
    for setting in setting_list:
        setting_val = EXT_OPH_SETTING_DICT[setting]
        for baseline in EXT_OPH_BASELINE:
            frame_list = EXT_OPH_EVAL_FRAME[baseline]
            frame_value = [FRAME_DICT[frame] for frame in frame_list]
            name_dict = EXT_OPH_EXPR_DEFAULT_NAME_DICT[baseline]
            out_folder = name_dict[0]
            expr = name_dict[1]
            for i, frame_val in enumerate(frame_value): # 3D, 2DCenter
                frame = frame_list[i] # 3D, 2D
                # print(name_dict)

                suffix = ""
                # print(out_folder)
                if (dataset_code, setting, out_folder, frame) in EXT_OPH_MISC_SUFFIX_DICT:
                    suffix = EXT_OPH_MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                    # print(dataset, setting, out_folder, frame, suffix)
                for fname in ["fold_results_test"]:
                    for ext in ["txt"]:
                        if (dataset_code, setting, out_folder) in EXT_OPH_MISC_FRAME_SUFFIX_DICT:
                            frame_val = EXT_OPH_MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                        file_path = ext_oph_get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext=ext)
                        # print(file_path)
                        print(f"Loading {file_path}")
                        ext_oph_load_fold_results(file_path)
                        try:
                            result = ext_oph_load_fold_results(file_path)
                            print(result)
                            ext_oph_results_dict[(dataset, setting, baseline, frame)] = result
                        except:
                            print(f"Error loading {file_path}")
                            continue 
    print("\n")

print("Results dict:", ext_oph_results_dict)
ext_oph_grouped_dict = get_ext_oph_task_and_setting_grouped_dict(ext_oph_results_dict)
