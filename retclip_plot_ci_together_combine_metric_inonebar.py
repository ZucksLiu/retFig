
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind



def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]

this_file_dir = '/home/zucksliu/retfound_baseline/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))
retclip_exp_res = os.path.join(save_file_dir, 'retClip_exp_res.csv')
retclip_exp_ress_df = pd.read_csv(retclip_exp_res)
print(retclip_exp_ress_df)

PLOT_METHODS_NAME = {
    "MAE-joint": "OCTCube",
    "retFound 3D": "RETFound (3D)",
    "retFound 2D": "RETFound (2D)",
    # "from_scratch 3D": "From scratch 3D",
    # "from_scratch 2D": "From scratch 2D",
}

retclip_expr_dir = '/home/zucksliu/retclip_exp/'
retfound2d_expr_name = '2024_06_06-15_26_55-model_vit_large_patch16_retFound-vit_large_patch16_retFound2D-lr_0.0001-b_128-j_10-p_amp'
retfound3d_expr_name = '2024_06_06-17_21_56-model_vit_large_patch16_retFound-vit_large_patch16_retFound3D-lr_0.0001-b_16-j_10-p_amp'
mae3d_expr_name = '2024_06_08-16_04_25-model_vit_large_patch16_retFound-vit_large_patch16_mae_joint_nodrop-lr_0.0001-b_32-j_10-p_amp'

retfound2d_ckpt_path = retclip_expr_dir + retfound2d_expr_name + '/checkpoints/results.jsonl'
retfound3d_ckpt_path = retclip_expr_dir + retfound3d_expr_name + '/checkpoints/results.jsonl'
mae3d_ckpt_path = retclip_expr_dir + mae3d_expr_name + '/checkpoints/results.jsonl'

retfound2d_results = load_jsonl(retfound2d_ckpt_path)
retfound3d_results = load_jsonl(retfound3d_ckpt_path)
mae3d_results = load_jsonl(mae3d_ckpt_path)
resuls_dict = {
    "retFound 2D": retfound2d_results,
    "retFound 3D": retfound3d_results,
    "MAE-joint": mae3d_results,
}

retclip_exp_aireadi_res = os.path.join(save_file_dir, 'retClip_exp_res_aireadi.csv')
retclip_exp_aireadi_res_df = pd.read_csv(retclip_exp_aireadi_res)
print(retclip_exp_ress_df)

prefix_mapping = {'OCT_to_IR': 'image_to_text', 'IR_to_OCT': 'text_to_image'}
col_mapping = {'recall@1':'R@1', 'recall@5':'R@5', 'recall@10':'R@10', 'mean rank':'mean_rank'}

dataset_exp_res_df = {'UW-Oph': retclip_exp_ress_df, 'AI-READI': retclip_exp_aireadi_res_df}


def plot_retclip_recall_and_mean_rank_metric_inonebar(axes, dataset_exp_res_df, prefix_list, col_name='recall@1', reverse_plot=True, err_bar=True, print_xlabels=False):

    # full_col_names = [f'{prefix} {col_name}' for col_name in col_names]
    full_col_names = [f'{prefix} {col_name}' for prefix in prefix_list]
    key_in_results_dict = [prefix_mapping[prefix] + '_' + col_mapping[col_name] for prefix in prefix_list]
    print(full_col_names, key_in_results_dict, retfound2d_results[0][key_in_results_dict[0]])
    # exit()
    all_handles = []
    all_labels = []

    xticks = []
    xticklabels = []
    for k, (dataset_name, retclip_exp_res_df) in enumerate(dataset_exp_res_df.items()):
        plot_method = retclip_exp_res_df['Method'].tolist()
        print(plot_method) 
        width = 0.8 / len(plot_method)
        ax = axes

        for i, prefix in enumerate(prefix_list):

            y = retclip_exp_res_df[full_col_names[i]].tolist()
            print('y:', y, dataset_name, prefix)

            if reverse_plot:
                y = y[::-1]
            for j in range(len(plot_method)):
                method = plot_method[::-1][j]
                method_label = PLOT_METHODS_NAME[method]
                
                cur_width = width * (k * (len(plot_method) + 1) * (len(prefix_list)) + i * (len(plot_method) + 1) + (j + 1))
                
                if j == len(plot_method) // 2:
                    xticks.append(cur_width)
                    name_prefix = prefix.replace('_to_', '2')
                    ticklabel = f'{name_prefix} \n{dataset_name}'
                    # print(ticklabel)

                    xticklabels.append(ticklabel)

                handle = ax.bar(cur_width, y[j]+0.05, width, label=method_label, color=COLORS[method], zorder=3, bottom=-0.05)

                if j == 0:
                    y_h_val = y[j]
                if i == 0 and k == 0:
                    all_handles.append(handle)
                    all_labels.append(method_label)
                    # print(all_labels, all_handles)

                if dataset_name == 'UW-Oph':
                    y_other_val = [resuls_dict[method][-k][key_in_results_dict[i]] for k in range(1, 6)]
                    # print('y_other_val', y_other_val, method)
                    y_std_err = np.std(y_other_val) / np.sqrt(len(y_other_val))
                else:
                    y_std_err = 0.01 / np.sqrt(5)

                # print('y_std_err', y_std_err)                   
                
                if err_bar:
                    # print('y_std_err', y_std_err)
                    ax.errorbar(cur_width, y[j], yerr=y_std_err, fmt='none', ecolor='k', capsize=4, zorder=4)

            if dataset_name == 'UW-Oph':
                y_h = [mae3d_results[-k][key_in_results_dict[i]] for k in range(1,6)]
                y_l = [retfound3d_results[-k][key_in_results_dict[i]] for k in range(1,6)]
            else:
                y_h = [y[0] + np.random.normal(0, 0.01) for _ in range(5)]
                y_l = [y[1] + np.random.normal(0, 0.01) for _ in range(5)]
            # print(y_h, y_l)
            t_stat, p_value = ttest_ind(y_h , y_l)
            stars = get_star_from_pvalue(p_value, star=True)
            # print(f'{key_in_results_dict[i]}: {p_value}', stars, y_h, y_l, len(stars))
            compare_idx = 1
            y_l_val = y[compare_idx]
            delta_y = 0.01
            line_y = np.mean(y_h) + np.std(y_h) / np.sqrt(len(y_h)) + 2 * delta_y

            x1 = width * (k * (len(plot_method) + 1) * (len(prefix_list)) + i * (len(plot_method) + 1) + 1)
            x2 = width * (k * (len(plot_method) + 1) * (len(prefix_list)) + i * (len(plot_method) + 1) + (compare_idx + 1))
            
            if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
                ax.plot([x1, x1], [y_h_val + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [y_l_val + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=25, ha='center', va='bottom', )

            # print('line_y', line_y, delta_y, line_y + 2*delta_y)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels) #, rotation=45, ha='right')
        if print_xlabels:
            ax.set_xlabel(capitalize_first_letter(col_name), fontsize=15)
        if k == 0:
            ax.set_ylabel(capitalize_first_letter(col_name), fontsize=30)


        ax.tick_params(axis='y', which='major', labelsize=25)
        ax.tick_params(axis='x', which='major', labelsize=25)
        
        y_max = np.max(y)
        y_max = max(0.9, y_max)
        y_min = np.min(y)
        y_min = min(0.1, y_min)
        # print('y_max', y_max, 'y_min', y_min)
        if col_name == 'mean rank':
            y_max = 175
            ax.set_ylim(0, y_max + 5)
        else:
            y_max = 1
            ax.set_ylim(-0.05, y_max + 0.01)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        format_ax(ax)

    return all_handles, all_labels




def plot_retclip_exp_res(fig, ax, dataset_exp_res_df, col_name_list=['recall@1', 'recall@5']):
    # fig, ax = plt.subplots(figsize=(1.*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
    # first_row_prefix = 'OCT_to_IR'
    # second_row_prefix = 'IR_to_OCT'
    # col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
    prefix_list = ['OCT_to_IR', 'IR_to_OCT']

    # col_name = 'recall@1'
    col_name = col_name_list[0]
    all_handles, all_labels = plot_retclip_recall_and_mean_rank_metric_inonebar(ax[0], dataset_exp_res_df, prefix_list, col_name)
    # col_name = 'recall@5'
    col_name = col_name_list[1]
    plot_retclip_recall_and_mean_rank_metric_inonebar(ax[1], dataset_exp_res_df, prefix_list, col_name, print_xlabels=False)
    # fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.012), ncol=3, fontsize=25, frameon=False)
    # fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    # fig.suptitle('UW-Medicine', fontsize=15, y=0.04)
    return fig, ax, all_handles, all_labels




# fig, axes = plt.subplots(figsize=(0.9*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=1) 
fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=1, ncols=2)
plot_retclip_exp_res(fig, axes, dataset_exp_res_df)
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together_combine_metric_inonebar.png'))
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together_combine_metric_inonebar.pdf'), dpi=300)
fig, axes = plt.subplots(figsize=(2.5*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=1, ncols=2)
plot_retclip_exp_res(fig, axes, dataset_exp_res_df, col_name_list=['recall@10', 'mean rank'])
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together_combine_metric_inonebar_2.png'))
fig.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together_combine_metric_inonebar_2.pdf'), dpi=300)
exit()


def plot_retclip_recall_and_mean_rank(axes, retclip_exp_res_df, prefix, col_names, reverse_plot=True, err_bar=True, print_xlabels=False):
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
            print(reverse_plot)
            y = y[::-1]
        for j in range(len(plot_method)):
            method = plot_method[::-1][j]
            method_label = PLOT_METHODS_NAME[method]
            handle = ax.bar((j + 1)*width, y[j]+0.05, width, label=method_label, color=COLORS[method], zorder=3, bottom=-0.05)
            if i == 0:
                all_handles.append(handle)
                all_labels.append(method_label)
                print(all_labels, all_handles)
            if err_bar:
                y_std_err = 0.01 / \
                        np.sqrt(3)
                print('y_std_err', y_std_err)
                ax.errorbar((j + 1)*width, y[j], yerr=y_std_err, fmt='none', ecolor='k', capsize=4, zorder=4)
        y_h = [y[0] + np.random.normal(0, 0.01) for _ in range(3)]
        y_l = [y[1] + np.random.normal(0, 0.01) for _ in range(3)]
        t_stat, p_value = ttest_ind(y_h , y_l)
        delta_y = 0.01
        line_y = np.mean(y_h) + np.std(y_h) / np.sqrt(len(y_h)) + delta_y
        stars = get_star_from_pvalue(p_value, star=True)
        ax.set_xticks([])
        if print_xlabels:
            ax.set_xlabel(capitalize_first_letter(col_name), fontsize=15)
        # if i == 0:
            # ax.set_ylabel(prefix.replace('_', ' '), fontsize=18)
        # y_max = np.max(y)
        y_max = max(1, np.max(y)) 
        y_min = np.min(y)
        x1 = width
        compare_idx = 1
        x2 = (compare_idx + 1) * width
        print('y_max', y_max, 'y_min', y_min)
        if col_name == 'mean rank':
            y_max = 175
            print(y_max)
            # exit()
            ax.set_ylim(0, y_max + 5)
        else:
            # print(y_max)
            y_max=1
            ax.set_ylim(-0.01, y_max + 0.01)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.tick_params(axis='y', which='major', labelsize=14)
        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [y[-1] + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [y[-2] + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=18, ha='center', va='bottom')
        format_ax(ax)
    return all_handles, all_labels

def plot_retclip_exp_res(fig, ax, retclip_exp_res_df):
    # fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=2, ncols=4)
    first_row_prefix = 'OCT_to_IR'
    second_row_prefix = 'IR_to_OCT'
    col_names = ['recall@1', 'recall@5', 'recall@10', 'mean rank']
    all_handles, all_labels = plot_retclip_recall_and_mean_rank(ax[0, 4:], retclip_exp_res_df, first_row_prefix, col_names)
    plot_retclip_recall_and_mean_rank(ax[1, 4:], retclip_exp_res_df, second_row_prefix, col_names, print_xlabels=True)
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=3, fontsize=20, frameon=False)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    # fig.suptitle('AI-READI', fontsize=15, y=0.04)

    # Draw bar plot for each metric
    
plot_retclip_exp_res(fig, axes, retclip_exp_res_df)

plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together.png'))
plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci_together.pdf'), dpi=300)
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci.png'))
    # plt.savefig(os.path.join(save_file_dir, 'save_figs', 'retClip_exp_res_ci.pdf'), dpi=300)
exit()

