
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *


# this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir = '~/retfound_baseline/'
EXP_CODE_DICT = {
                'Prov-GigaPath': 'GigaPath',
                #r'HIPT-$Prov$-$Path$': 'HIPT_(Ours)',
                'HIPT': 'HIPT_(Public)',
                'CtransPath': 'CtransPath',
                'REMEDIS': 'REMEDIS',
            }

TASKS = {
    'Prov170k-All-25-gene': 'Pan 18-biomarker',
    'Prov170k-LUAD-5-gene': 'LUAD 5-Gene',
    'Prov170k-All-5-gene': 'Pan 5-Gene',
    'TCGA-LUAD-5-gene': 'LUAD 5-Gene (TCGA)',
    'Prov170k-TMB-Mutation': f'Pan TMB',
    'Prov170k-EGFR-Mutation': f'Pan ' + r'$EFGR$' + ' subvariants',
}

boxprops = {'edgecolor': 'k'}
whiskerprops = {'color': 'k'}
capprops = {'color': 'k'}
medianprops = {'color': 'r', 'linewidth': 1.5}
  

if __name__ == '__main__':
    
    # make save dir
    os.makedirs(os.path.join(this_file_dir, 'save_figs'), exist_ok=True)
    
    # set up the figure axes
    fig, axes = plt.subplots(figsize=(1*FIG_WIDTH, 0.7*FIG_HEIGHT), nrows=3, ncols=5)

    results = {}
    for task in TASKS:
        
        task_root_dir = os.path.join(this_file_dir, '../', f'Results/{task}')
        df_dict = {}
        for exp_code in EXP_CODE_DICT.keys():
            df_dict[exp_code] = pd.read_csv(os.path.join(task_root_dir, EXP_CODE_DICT[exp_code] + '.csv'))
        results[TASKS[task]] = df_dict
    
    # plot the subfigure a-e
    mutation_5_tasks_barplot_fixed_range(axes[0, :], results, 'macro_auroc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUROC', y_min=0.4, y_max=0.75, plot_xlabel=True)
    # plot the subfigure f-j
    mutation_5_tasks_barplot_fixed_range(axes[1, :], results, 'macro_auprc', list(TASKS.values()), list(EXP_CODE_DICT.keys()), 'AUPRC', y_min=0.0, y_max=0.45)
    # plot the subfigure k
    mutation_5_gene_each_class_barplot_fixed_range(axes[2, :], results, y_min=0.25, y_max=0.82)

    fig.tight_layout()
    plt.savefig(os.path.join(this_file_dir, 'save_figs', 'figure_2a-k.pdf'), dpi=300)

    fig, ax = plt.subplots(figsize=(1*FIG_WIDTH, 0.35*FIG_HEIGHT))
    mutation_18_biomarker_each_class_plot(ax, results, plot_title=False)

    fig.tight_layout()
    plt.savefig(os.path.join(this_file_dir, 'save_figs', 'figure_2l.pdf'), dpi=300)