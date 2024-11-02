
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind


this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))
crossmodal_results_dir = os.path.join(save_file_dir, 'CrossModal-results-20241027')

CROSSMODAL_EXPR_NAME = {
    'CT3D': (['OCTCube', 'ConvNext-SLIViT', 'RETFound'], ['AUC', 'AUCPR']),
    'EF': (['OCTCube-SLIViT', 'ConvNext-SLIViT', 'RETFound'], ['R2']),
    'EF_b': (['OCTCube', 'ConvNext-SLIViT', 'RETFound'], ['AUC', 'AUCPR']),
}
all_plot_metrics = sorted(set([metric for method_list, metric_list in CROSSMODAL_EXPR_NAME.values() for metric in metric_list]))
all_plot_metrics = ['AUCPR', 'AUC', 'R2']
print('All metric that will be plot:', all_plot_metrics)


def load_expr_results(crossmodal_results_dir, expr_name):
    expr_results = {}
    method_list, metric_list = CROSSMODAL_EXPR_NAME[expr_name]
    for method in method_list:
        df = pd.read_csv(os.path.join(crossmodal_results_dir, f'{expr_name}/{method}.csv'))
        expr_results[method] = df
    return expr_results
print('load_expr_results')
CT3D_results = load_expr_results(crossmodal_results_dir, 'CT3D')
EF_results = load_expr_results(crossmodal_results_dir, 'EF')
EF_b_results = load_expr_results(crossmodal_results_dir, 'EF_b')
print(CT3D_results)
print(EF_results)
print(EF_b_results)
plot_dict = {
    'CT3D': CT3D_results,
    'EF': EF_results,
    'EF_b': EF_b_results,
}

PLOT_X_LABELS = {
    'CT3D': 'CT (3D)',
    'EF': 'US EF (2D+T)',
    'EF_b': 'US EF_b (2D+T)',
}
PLOT_METHODS_NAME = {
    'OCTCube-SLIViT': "OCTCube",
    "OCTCube": "OCTCube",
    "ConvNext-SLIViT": "SLIViT",
    "RETFound": "RETFound",
}

PLOT_METRIC_NAME = {
    'AUC': 'AUROC',
    'AUCPR': 'AUPRC',
    'R2': r'$R^2$',
}


def organize_expr_by_metric(plot_dict, metric_name):
    '''
    Organize the plot_dict by metric_name,
    Example:
            metric_name = 'AUC'
            auc_plot_dict = organize_expr_by_metric(plot_dict, metric_name)
            print(auc_plot_dict)
    '''
    organized_dict = {}
    for expr_name, expr_results in plot_dict.items():
        
        for method, df in expr_results.items():
            print(method, df)
            if metric_name not in df.columns:
                continue
            else:
                if expr_name not in organized_dict:
                    organized_dict[expr_name] = {}
            organized_dict[expr_name][method] = df[metric_name].tolist()
    return organized_dict

bymetric_plot_dict = {}
for metric_name in all_plot_metrics:
    bymetric_plot_dict[metric_name] = organize_expr_by_metric(plot_dict, metric_name)
    print(bymetric_plot_dict[metric_name])

    


def plot_slivit_crossmodal_metric_inonebar(axes, dataset_exp_res_df, metric_name, plot_task=None, plot_method=None, color_selection=None, reverse_plot=False, err_bar=True, print_xlabels=False, print_ylabels=True):
    if plot_task is None:
        plot_task = list(dataset_exp_res_df.keys())
    print('plot_task:', plot_task)
    
    if plot_method is None:
        plot_method = list(dataset_exp_res_df[plot_task[0]])
    print('plot_method:', plot_method)

    if color_selection is None:
        color_mapping = [COLORS_SLIVIT_CROSSMODAL_compute[m] for m in plot_method]
    else:
        color_mapping = color_selection

    print('color_mapping:', color_mapping)

    # cohort_list = dataset_exp_res_df.columns.tolist()
    # cohort_list.remove('Method')
    # print(cohort_list)
    # pval_dict, pval_method_name_dict = metric_mapping[metric_name]
    plot_metric_name = PLOT_METRIC_NAME[metric_name]
    print('plot_metric_name:', plot_metric_name)

    # print(pval_dict, pval_method_name_dict)
    # exit()
    # pval_list = [pval_dict[cohort] for cohort in cohort_list]
    # pval_method_idx = [plot_method.index(pval_method_name_dict[cohort]) for cohort in cohort_list]
    # print(pval_dict, pval_method_idx)
    # exit()
    all_handles = []
    all_labels = []
    width = 0.8 / len(plot_method)
    ax = axes
    xticks = []
    xticklabels = []
    # dataset_exp_res_df_used = dataset_exp_res_df[dataset_exp_res_df['Method'].isin(plot_method)]
    # dataset_exp_res_std_df_used = dataset_exp_res_std_df[dataset_exp_res_std_df['Method'].isin(plot_method)]
    # print(dataset_exp_res_df_used)
    print(dataset_exp_res_df)
    

    for i, task_i in enumerate(plot_task):
        y_method = dataset_exp_res_df[task_i]
        
        print('y_method:', y_method)
        if metric_name == 'AUCPR':
            y_l_idx = 2
        else:
            y_l_idx = 1

        for j in range(len(plot_method)):
            method = plot_method[j]
            method_label = PLOT_METHODS_NAME[method]
            
            cur_width = width * (i * (len(plot_method) + 1) + (j + 1))
            y_method_j = y_method[method]
            print('method_name', method, 'y_method_j:', y_method_j, 'method_label:', method_label)
            y = np.array(y_method_j)
            y_mean = np.mean(y)
            y_std_err = np.std(y) / np.sqrt(len(y))
            print('y_mean:', y_mean, 'y_std_err:', y_std_err, 'y:', y)


            if j == len(plot_method) // 2:
                xticks.append(cur_width)
                ticklabel = PLOT_X_LABELS[task_i]
                xticklabels.append(ticklabel)

            handle = ax.bar(cur_width, y_mean+0.05, width, label=method_label, color=color_mapping[j], zorder=3, bottom=-0.05)

            if j == 0:
                y_h = y
                y_h_val = y_mean
                y_std_err_h = y_std_err
            if j == y_l_idx:
                y_l = y
                y_l_val = y_mean
                y_std_err_l = y_std_err
            if i == 0:
                all_handles.append(handle)
                all_labels.append(method_label)
                

            # if dataset_name == 'UW-Oph':
            #     y_other_val = [resuls_dict[method][-k][key_in_results_dict[i]] for k in range(1, 6)]
            #     # print('y_other_val', y_other_val, method)
            #     y_std_err = np.std(y_other_val) / np.sqrt(len(y_other_val))
            # else:
            # if True:
                # y_std_err = np.random.rand() * 0.002 / np.sqrt(5)
                # y_std_err = std_list[j]               
            
            if err_bar:
                # print('y_std_err', y_std_err)
                ax.errorbar(cur_width, y_mean, yerr=y_std_err, fmt='none', ecolor='k', capsize=4, zorder=4)

        # if dataset_name == 'UW-Oph':
        #     y_h = [mae3d_results[-k][key_in_results_dict[i]] for k in range(1,6)]
        #     y_l = [retfound3d_results[-k][key_in_results_dict[i]] for k in range(1,6)]
        # else:
        #     y_h = [y[0] + np.random.normal(0, 0.01) for _ in range(5)]
        #     y_l = [y[1] + np.random.normal(0, 0.01) for _ in range(5)]

        print('y_h and y_l:', y_h, y_l)
        # t_stat, p_value = ttest_ind(y_h , y_l)
        t_stat, p_value = ttest_rel(y_h , y_l)
        print('t_stat:', t_stat, 'p_value:', p_value)
        pval_list = [p_value]
        pval_method_idx_list = [y_l_idx]
        delta_list = [0.01]

        for p_idx, p_value in enumerate(pval_list):
            # p_value = pval_list[i]
            stars = get_star_from_pvalue(p_value, star=True)
            print('stars:', stars, p_value)
            # pval_method_name = pval_method[p_idx]
            # pval_method_idx = plot_method.index(pval_method_name)
            pval_method_idx = pval_method_idx_list[p_idx]
            # y_h_val = y[0]
            # y_l_val = y[pval_method_idx]    
            # y_std_err_h = std_list[0]
            # y_std_err_l = std_list[pval_method_idx]

            # print(pval_method_idx, y_h_val, y_l_val)
            delta_y = delta_list[p_idx]
            line_y = y_h_val + y_std_err_h + 2 * delta_y

            x1 = width * (i * (len(plot_method) + 1) + 1)
            x2 = width * (i * (len(plot_method) + 1) + (pval_method_idx + 1))
        
            if np.mean(y_h_val) > np.mean(y_l_val) and len(stars) > 0:
                ax.plot([x1, x1], [y_h_val + y_std_err_h + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [y_l_val + y_std_err_l + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=25, ha='center')#, va='bottom', )
        if i == 0 and print_ylabels:
            ax.set_ylabel(f'{plot_metric_name}', fontsize=25)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels) #, rotation=45, ha='right')
    # if print_xlabels:
    #     ax.set_xlabel(capitalize_first_letter('Test'), fontsize=15)



    ax.tick_params(axis='y', which='major', labelsize=25)
    ax.tick_params(axis='x', which='major', labelsize=25)
    
    y_max = np.max(y)
    y_max = max(0.77, y_max)
    y_min = np.min(y)

    y_min = min(0.6, y_min)
    if metric_name == 'R2':
        y_max = 0.8
        y_min = 0.4
    print('y_max', y_max, 'y_min', y_min)
    # if col_name == 'mean rank':
    #     y_max = 175
    #     ax.set_ylim(0, y_max + 5)
    # else:
    if True:
        # y_max = 1
        ax.set_ylim(0.6, y_max + 0.01)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        if metric_name == 'R2':
            ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8])
            ax.set_ylim(0.4, 0.8)

    # ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
    format_ax(ax)
    # full_col_names = [f'{prefix} {col_name}' for col_name in col_names]
    # print(full_col_names)
    # exit()
    # full_col_names = [f'{prefix} {col_name}' for col_name in col_names]
    # full_col_names = [f'{prefix} {col_name}' for prefix in prefix_list]
    # key_in_results_dict = [prefix_mapping[prefix] + '_' + col_mapping[col_name] for prefix in prefix_list]
    # print(full_col_names, key_in_results_dict, retfound2d_results[0][key_in_results_dict[0]])
    # exit()


    return all_handles, all_labels




def plot_slivit_crossmodal_exp_res(fig, ax, dataset_exp_res_df, plot_metric=None, plot_method=None, color_selection=None, legend_ncol=4, bbox_adjust_ratio=0.9):
    '''
    Plot the SLIViT crossmodal data classification results
    dataset_exp_res_df: a list of dataframes, each dataframe contains the results of a dataset, should be by metric
    '''
    # fig, ax = plt.subplots(figsize=(1.*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
    # first_row_prefix = 'OCT_to_IR'
    # second_row_prefix = 'IR_to_OCT'
    # col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
    # prefix_list = ['OCT_to_IR', 'IR_to_OCT']

    # col_name = 'recall@1'
    # col_name = col_name_list[0]
    print(dataset_exp_res_df)
    print(dataset_exp_res_df.keys(), dataset_exp_res_df['AUC'].keys())
    if plot_metric is None:
        plot_metric = list(dataset_exp_res_df.keys())
    for i in range(len(plot_metric)):
        all_handles, all_labels = plot_slivit_crossmodal_metric_inonebar(ax[i], dataset_exp_res_df[plot_metric[i]], plot_metric[i]) #, plot_method=plot_method, color_selection=color_selection)
    # exit()
    # all_handles, all_labels = plot_retclip_recall_and_mean_rank_metric_inonebar(ax[0], dataset_exp_res_df[0], dataset_exp_res_std_df[0], 'BCVA', print_ylabels=True, plot_method=plot_method, color_selection=color_selection)
    # # col_name = 'recall@5'
    # # col_name = col_name_list[1]
    # plot_retclip_recall_and_mean_rank_metric_inonebar(ax[1], dataset_exp_res_df[1], dataset_exp_res_std_df[1], 'GAGrowth', print_xlabels=False, print_ylabels=True, plot_method=plot_method, color_selection=color_selection)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=legend_ncol, fontsize=25, frameon=False)
    # fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.tight_layout(rect=[0, -0.02, 1, bbox_adjust_ratio])
    # fig.suptitle('UW-Medicine', fontsize=15, y=0.04)
    return fig, ax, all_handles, all_labels



# dataset_exp_res_df = [GA_prog_BCVA_mean_res_df, GA_prog_GAGrowth_mean_res_df]
# dataset_exp_res_std_df = [GA_prog_BCVA_std_res_df, GA_prog_GAGrowth_std_res_df]
# plot_methods_used = ['OCTCube-COEP-3mod', 'OCTCube-smOCT', 'FAF-DenseNet', 'OCT-DenseNet3D', 'RETFound-smOCT']

COLORS_SLIVIT_CROSSMODAL_compute

# color_selection = [COLORS_GA_prog_compute[m] for m in plot_methods_used]
dataset_exp_res_df = bymetric_plot_dict
fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=1, ncols=3, gridspec_kw={'width_ratios': [2, 2, 1]})
plot_slivit_crossmodal_exp_res(fig, axes, dataset_exp_res_df=dataset_exp_res_df)#, dataset_exp_res_std_df=dataset_exp_res_std_df)#, plot_method=plot_methods_used,  bbox_adjust_ratio=0.9) # color_selection=color_selection,
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'SLIViT_crossmodal_data_cls_plot_top5_allinonebar_moreRes.png'), dpi=300)
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'SLIViT_crossmodal_data_cls_plot_top5_allinonebar_moreRes.pdf'), dpi=300)


# fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=1, ncols=2)
# plot_retclip_exp_res(fig, axes, dataset_exp_res_df=dataset_exp_res_df)
# fig.savefig(os.path.join(save_file_dir, 'save_figs', 'GA_prog_BCVA_mean_res_ci_together_combine_metric_inonebar.png'))
# fig.savefig(os.path.join(save_file_dir, 'save_figs', 'GA_prog_BCVA_mean_res_ci_together_combine_metric_inonebar.pdf'), dpi=300)

exit()






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
            handle = ax.bar((j + 1)*width, y[j]+0.05, width, label=method_label, color=COLORS[method], zorder=3, bottom=-0.05)
            if i == 0:
                all_handles.append(handle)
                all_labels.append(method_label)
                print(all_labels, all_handles)
        ax.set_xticks([])
        ax.set_xlabel(capitalize_first_letter(col_name), fontsize=8)
        if i == 0:
            ax.set_ylabel(prefix.replace('_', ' '), fontsize=8)
        # y_max = np.max(y)
        y_max = max(1, np.max(y)) 
        y_min = np.min(y)
        print('y_max', y_max, 'y_min', y_min)
        if col_name == 'mean rank':
            ax.set_ylim(0, y_max + 5)
        else:
            ax.set_ylim(-0.01, y_max + 0.01)
        format_ax(ax)
    return all_handles, all_labels




def plot_GA_prog_exp_res(GA_prog_exp_res_df):
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
    first_row_prefix = 'OCT_to_IR'
    second_row_prefix = 'IR_to_OCT'
    col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
    all_handles, all_labels = plot_retclip_recall_and_mean_rank(ax[0, :], GA_prog_exp_res_df, first_row_prefix, col_names)
    plot_retclip_recall_and_mean_rank(ax[1, :], GA_prog_exp_res_df, second_row_prefix, col_names)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.012), ncol=3, fontsize=8, frameon=False)
    fig.suptitle('GA BL BCVA', fontsize=10, y=0.03)
    fig.tight_layout()
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'GA_prog_exp_res.png'))
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'GA_prog_exp_res.pdf'), dpi=300)
    # Draw bar plot for each metric
    
plot_GA_prog_exp_res(GA_prog_BCVA_mean_res_df)
exit()




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
BASELINE = ["MAE-joint", "retfound",  "from_scratch", ] # "imagenet",
FRAME_DICT = {"3D": "3D", "2D": "2DCenter"}
SETTING_DICT = {"fewshot": "correct_patient_fewshot", "default": "correct_patient"}

# Baseline method and the corresponding avialable dimensional settings
INHOUSE_EVAL_FRAME = {
    "MAE-joint": ["3D"],
    "imagenet": ["3D"],
    "retfound": ["3D", "2D"],
    "from_scratch": ["3D", "2D"],

} 

EXPR_DEFAULT_NAME_DICT = {
    "retfound 2D": ["outputs_ft_0529_ckpt_flash_attn", "retfound"],
    "from_scratch 2D": ["outputs_ft_0529_ckpt_flash_attn", "no_retfound"],
    "retfound 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_retfound"],
    "from_scratch 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_no_retfound"],
    "MAE-joint 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_3d_256_0509"],
    "imagenet 3D": ["outputs_ft_st_0529_ckpt_flash_attn", "singlefold_3d_imagenet"],
}

# Exteneded baseline methods with dimensionality and the plotting method name
PLOT_METHODS_NAME = {
    "MAE-joint 3D": "OCTCube",
    "retfound 3D": "RETFound (3D)",
    "from_scratch 3D": "Supervised (3D, w/o pre-training)",
    "retfound 2D": "RETFound (2D)",
    "from_scratch 2D": "Supervised (2D, w/o pre-training)",
    "imagenet 3D": "ImageNet trasferred 3D MAE",
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
}

# Miscellaneous frame suffix dictionary for the output folder
MISC_FRAME_SUFFIX_DICT = {
    # ("hcms", "fewshot", "outputs_ft_st"): { "3D": "3D_st"},
    # ("hcms", "default", "outputs_ft_st"): { "3D": "3D_st"}
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


def get_results_json(out_dir, prefix='finetune_inhouse_', dataset="", frame="", setting="", expr="", suffix="", fname="", ext="json"):
    if setting != "":
        setting = f"_{setting}"
    if expr != "":
        expr = f"_{expr}"
    if suffix != "":
        expr = f"{expr}_{suffix}"
        # print(expr, suffix)
    return out_dir + f"{prefix}{dataset}_{frame}{setting}{expr}/{fname}.{ext}"


def get_inhouse_task_and_setting_grouped_dict(results_dict, results_csv_dict):
    setting_list = SETTING_DICT.keys()
    task_list = INHOUSE_OPH_DATASET_DICT.keys()
    grouped_dict = {}
    for setting in setting_list:
        setting_val = SETTING_DICT[setting]
        grouped_dict[setting] = {}
        for task in task_list:
            task_code = INHOUSE_OPH_DATASET_DICT[task]
            available_setting = INHOUSE_OPH_DATASET_EVAL_SETTING[task]
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


def INHOUSE_oph_tasks_barplot(fig, axes, grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], plot_methods_name=None, y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1], err_bar=True):
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
        cur_top5_val = []
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            print(j, m)
            result, result_csv = df_dict[plot_task][m]
            y = np.mean(result[:, plot_col_idx])
            print(result_csv.columns)
            plot_col_csv = find_candidate_metric(plot_col, result_csv)
            if plot_col_csv is None:
                print(f"Cannot find {plot_col} in {result_csv.columns}")
                exit()
            
            y_other_val = result_csv[plot_col_csv]
            for idx, val in enumerate(y_other_val):
                try:
                    y_other_val[idx] = float(val)
                except:
                    y_other_val[idx] = 0.0
            try:
                y_other_val = y_other_val.astype(float)
            except:
                print('Exist error in converting y_other_val to float')
                exit()
            print('y_other_val before error', y_other_val, type(y_other_val))
            y_other_val_top5 = list(y_other_val.nlargest(5))
            print(y, 'y_other_val', y_other_val, y_other_val_top5, type(y_other_val_top5)) 
            cur_top5_val.append(y_other_val_top5)
            # exit()
            handle = ax.bar(i * width * (len(plot_methods) + 1) + (j + 1) * width, y, width, label=plot_methods_name[j], color=COLORS[plot_methods_name_key[j]], zorder=3)
            if err_bar:
                y_std_err = np.std(y_other_val_top5) / \
                        np.sqrt(len(y_other_val_top5))
                ax.errorbar(i * width * (len(plot_methods) + 1) + (j + 1) * width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)

            if y > best_y and j != 0:
                best_y = y
                compare_col = m
                compare_col_idx = j
            if i == 0:  # Collect handle for legend only once per method across all tasks
                all_handles.append(handle)
                all_labels.append(plot_methods_name[j])

        print(df_dict[plot_task][plot_methods[0]])
        agg_ours.append(np.mean(df_dict[plot_task][plot_methods[0]][0][:, plot_col_idx]))
        agg_r3d.append(np.mean(df_dict[plot_task][plot_methods[3]][0][:, plot_col_idx]))

        print('agg_ours:', agg_ours, 'agg_r3d:', agg_r3d)

        y_min = np.min([np.mean(df_dict[plot_task][m][0][:, plot_col_idx]) for m in plot_methods])
        if plot_col == 'auroc':
            y_min = np.min([y_min, 0.5])
        elif plot_col == 'auprc':
            y_min = np.min([y_min, 0.5]) 
        # y_min = np.min([list(df_dict[plot_task][m][:, plot_col_idx]) for m in plot_methods])
        y_max_temp = np.max([np.mean(df_dict[plot_task][m][0][:, plot_col_idx]) + np.std(cur_top5_val[m_idx]) / \
                        np.sqrt(len(cur_top5_val[m_idx])) for m_idx, m in enumerate(plot_methods)])
        if y_max_temp > y_max:
            y_max = y_max_temp

        y_h = cur_top5_val[0] # df_dict[plot_task][plot_methods[0]][:, plot_col_idx].tolist()
        y_l = cur_top5_val[compare_col_idx] # df_dict[plot_task][compare_col][:, plot_col_idx].tolist()
        
        # p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        t_stat, p_value = ttest_ind(y_h * 2 , y_l *2)
        # print(compare_col, plot_methods_name[0], p_value)

        xticks_list.append(i * width * (len(plot_methods) + 1) + width * (len(plot_methods) + 1) / 2)
        xticks_label.append(plot_task)
        # ax.set_xlabel(plot_task)
        if i == 0:
            ax.set_ylabel(y_name, fontsize=15)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value, star=True)
        print(f'{plot_task}: {p_value}', stars, y_h, y_l, len(stars))
        compare_idx = plot_methods.index(compare_col)
        line_y = agg_ours[i] + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = i * width * (len(plot_methods) + 1) + width
        x2 = i * width * (len(plot_methods) + 1) + (compare_idx + 1)*width
        
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [agg_ours[i] + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=15, ha='center', va='bottom')
        format_ax(ax)
        print('line_y', line_y, delta_y, line_y + 2*delta_y)
        if y_max_line < line_y + 1*delta_y:
            y_max_line = line_y + 1*delta_y
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label, fontsize=15)
    ax.set_ylim(floor_to_nearest(y_min, 0.004), y_max_line)
    ax.set_xlim(0.05, 7.8)  # Adjust the first number to increase/decrease space
    ax.tick_params(axis='y', which='major', labelsize=10)

    avg_ours = np.mean(agg_ours)
    avg_r3d = np.mean(agg_r3d)
    avg_improvement = avg_ours - avg_r3d
    avg_rel_improvement = avg_improvement / avg_r3d
    print(f'{plot_col}, Average improvement: {avg_improvement}, Average relative improvement: {avg_rel_improvement}', 'avg_ours:', avg_ours, 'avg_r3d:', avg_r3d)
    return all_handles, all_labels
    # add legend for the axes
    


if __name__ == '__main__':
    results_dict = {}
    results_csv_dict = {}
    for dataset, dataset_code in INHOUSE_OPH_DATASET_DICT.items():
        setting_list = INHOUSE_OPH_DATASET_EVAL_SETTING[dataset] 
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
                    if (dataset_code, setting, out_folder, frame) in MISC_SUFFIX_DICT:
                        suffix = MISC_SUFFIX_DICT[(dataset_code, setting, out_folder, frame)]
                        # print(dataset, setting, out_folder, frame, suffix)
                    
                    # for fname in ["metrics_test_singlefold"]:
                    #     # for ext in ["csv"]:
                    fname = 'metrics_test_singlefold'
                    replace_fname = 'macro_metrics_test_singlefold'
                    if (dataset_code, setting, out_folder) in MISC_FRAME_SUFFIX_DICT:
                        frame_val = MISC_FRAME_SUFFIX_DICT[(dataset_code, setting, out_folder)][frame]
                    file_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext='csv')
                    try:
                        result = pd.read_csv(file_path) 
                        results_csv_dict[(dataset, setting, baseline, frame)] = result
                    except:
                        print(f"Error loading {file_path}")
                        replace_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='csv')
                        print(f"Loading {replace_path}")
                        result = pd.read_csv(replace_path) # result = load_fold_results(replace_path)
                        print(result)
                        results_csv_dict[(dataset, setting, baseline, frame)] = result
                    # print(f"Loading {file_path}")
                    fname = 'fold_results_test'
                    replace_fname = 'fold_results_test_for_best_val'
                    file_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=fname, ext='txt')
                    try:
                        result = load_fold_results(file_path)
                        print(result)
                        results_dict[(dataset, setting, baseline, frame)] = result
                    except:
                        print(f"Error loading {file_path}")
                        replace_path = get_results_json(this_file_dir + out_folder + '/', dataset=dataset_code, frame=frame_val, setting=setting_val, expr=expr, suffix=suffix, fname=replace_fname, ext='txt')
                        print(f"Loading {replace_path}")
                        result = pd.read_csv(replace_path) # result = load_fold_results(replace_path)
                        print(result)
                        results_dict[(dataset, setting, baseline, frame)] = result
                        continue 
        print("\n")
    print(results_csv_dict)

    print(results_dict)
    grouped_dict = get_inhouse_task_and_setting_grouped_dict(results_dict, results_csv_dict)
    print(grouped_dict)
    # exit()

    # Plot the figure
    fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=1)

    # results = {}
    # for task in TASKS:
        
    #     task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
    #     df_dict = {}
    #     for exp_code in EXP_CODE_DICT.keys():
    #         df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
    #     results[TASKS[task]] = df_dict

    # plot the subfigure a-e
    INHOUSE_oph_tasks_barplot(fig, axes[0], grouped_dict, setting_code='fewshot', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC') # auprc, Average improvement: 0.07385606829182212, Average relative improvement: 0.09980902644467299 avg_ours: 0.8138299055555556 avg_r3d: 0.7399738372637334
    # auprc, Average improvement: 0.03671960444444444, Average relative improvement: 0.04725147046943376 avg_ours: 0.8138299055555556 avg_r3d: 0.7771103011111111
    import time
    # time.sleep(10)
    # plot the subfigure f-j
    all_handles, all_labels = INHOUSE_oph_tasks_barplot(fig, axes[1], grouped_dict, setting_code='fewshot', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC') # auroc, Average improvement: 0.0778715390164183, Average relative improvement: 0.09635756177798718 avg_ours: 0.8860233599999999 avg_r3d: 0.8081518209835816 # auroc, Average improvement: 0.028492033333333278, Average relative improvement: 0.03322564721231286 avg_ours: 0.8860233599999999 avg_r3d: 0.8575313266666666

    
    # INHOUSE_oph_tasks_barplot(fig, axes[2, :], grouped_dict, setting_code='fewshot', plot_col='bal_acc', plot_tasks=[], plot_methods=[], y_name='BALANCED_ACC')
    # mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    # mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.5, w_pad=0.5)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=9, fontsize=15, columnspacing=0.8, frameon=False)
    # fig.tight_layout()

    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3a-k_ci_allinonebar.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3a-k_ci_allinonebar.png'))
    import time
    # time.sleep(10)    
    fig, axes = plt.subplots(figsize=(2*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=1)
    INHOUSE_oph_tasks_barplot(fig, axes[0], grouped_dict, setting_code='default', plot_col='auprc', plot_tasks=[], plot_methods=[], y_name='AUPRC') # auprc, Average improvement: 0.05084222170399999, Average relative improvement: 0.06519408308309639 avg_ours: 0.8307016705928889 avg_r3d: 0.7798594488888889 # auprc, Average improvement: 0.03057266149110005, Average relative improvement: 0.03820966512065398 avg_ours: 0.8307016705928889 avg_r3d: 0.8001290091017889
    import time
    # time.sleep(10)    
    INHOUSE_oph_tasks_barplot(fig, axes[1], grouped_dict, setting_code='default', plot_col='auroc', plot_tasks=[], plot_methods=[], y_name='AUROC')  # auroc, Average improvement: 0.0386585817663877, Average relative improvement: 0.04523032394557643 avg_ours: 0.8933635317663876 avg_r3d: 0.8547049499999999 # auroc, Average improvement: 0.02140687570578148, Average relative improvement: 0.024550389697688798 avg_ours: 0.8933635317663876 avg_r3d: 0.8719566560606061
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=0.5)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.015), ncol=9, fontsize=15, columnspacing=0.8, frameon=False)

    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l_ci_allinonebar.pdf'), dpi=300)
    plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_3l_ci_allinonebar.png'))
    

    # mutation_18_biomarker_each_class_plot(ax, results, plot_title=False)

    # fig.tight_layout()
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'figure_2l.pdf'), dpi=300)


    # auprc, Average improvement: 0.284830511403253, Average relative improvement: 0.5384325852767391 avg_ours: 0.8138299055555556 avg_r3d: 0.5289993941523026
    # auroc, Average improvement: 0.2768955093539056, Average relative improvement: 0.45457699735811785 avg_ours: 0.8860233599999999 avg_r3d: 0.6091278506460943
    # auprc, Average improvement: 0.29582335075009003, Average relative improvement: 0.5530666317472593 avg_ours: 0.8307016705928889 avg_r3d: 0.5348783198427989
    # auroc, Average improvement: 0.27405964708014185, Average relative improvement: 0.44252854512447803 avg_ours: 0.8933635317663876 avg_r3d: 0.6193038846862458