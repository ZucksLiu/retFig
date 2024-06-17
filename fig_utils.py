import json
import math
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter

from scipy import stats
from matplotlib.colors import to_rgba
from fig_settings import *
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.statistics import multivariate_logrank_test
from scipy.stats import wilcoxon

def capitalize_first_letter(s):
    return s[0].upper() + s[1:]

def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def format_ax(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

 
def format_y_tick(ax, y_max, y_min):
    # get y tick label from the y_min and y_max, make sure only keep one decimal
    y_ticks = []
    y_tick_s = np.ceil(y_min*10)/10
    while y_tick_s < y_max:
        y_ticks.append(y_tick_s)
        y_tick_s += 0.1
    ax.set_yticks(y_ticks)


def ceil_to_nearest(x, base=0.02):
    return math.ceil(x / base) * base


def floor_to_nearest(x, base=0.02):
    return math.floor(x / base) * base


def format_legend(fig_legend, handles, labels, legendmarker=20, loc='center', ncols=1, **kwargs):
    return fig_legend.legend(handles, labels, loc=loc, scatterpoints=1, ncol=ncols,
                      frameon=False, markerscale=legendmarker, **kwargs)
    

def add_bar_plot_dots(ax, x, y, s=5, jitter=0.1):
    np.random.seed(42)
    # noise is uniform distribution between -0.5 and 0.5
    noise = np.random.rand(len(x)) - 0.5
    x += jitter*noise
    ax.scatter(x, y, s=s, marker='o', linewidths=0.5,
               edgecolors='#525252', facecolors='none', zorder=5)


def add_box_plot_dots(ax, x, y, jitter=0.1, color='k'):
    np.random.seed(42)
    noise = np.random.randn(len(x))
    x += jitter*noise
    ax.scatter(x, y, c=color, \
            s=get_box_plot_setting()['marker size'], \
            marker='o', edgecolors=ax.spines['bottom'].get_edgecolor(),
            zorder=5)


def box_plot(ax, values, methods, colors):
    boxplots = ax.boxplot(values.T, patch_artist = True, showfliers=False)
    for patch, color in zip(boxplots['boxes'], colors):
        rgba_color = to_rgba(color, alpha=get_box_plot_setting()['box alpha'])
        patch.set_facecolor(rgba_color)
        patch.set_linewidth(get_box_plot_setting()['box linewidth'])
    for median in boxplots['medians']:
        median.set(color=get_box_plot_setting()['median color'],
                linewidth=get_box_plot_setting()['median linewidth'])
    for whisker in boxplots['whiskers']:
        whisker.set(linewidth=get_box_plot_setting()['whisker linewidth'],
                    linestyle=get_box_plot_setting()['whisker linestyle'])
    for cap in boxplots['caps']:
        cap.set(linewidth=get_box_plot_setting()['cap linewidth'])
    for i in range(len(methods)):
        dots_x, dots_y = [], []
        for j in range(np.size(values, 1)):
            dots_x.append(i + 1)
            dots_y.append(values[i, j])
        add_box_plot_dots(ax, dots_x, dots_y, jitter=0.12, color=colors[i])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def get_star_from_pvalue(p_value, star=False):
    if star:
        if p_value == 0:
            stars = ''
        elif p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'
        else:
            stars = ''
    else:
        # plot the number
        if p_value < 0.001:
            stars = r'$P$ < 0.001'
        else:
            stars = r'$P$ = {:.3f}'.format(p_value)
    return stars
 
 
# Calculate the statistical significance and determine how many stars to use
def add_significance_symbol_barplot(ax, methods, mean, err, our_method_index=0, n_fold=5):
    '''
    the size of mean is [n_group, n_method]
    '''
    n_group = np.size(mean, 0)
    width = 1. / len(methods) * 0.8
    for i in range(n_group):
        mean_exd = mean.copy()
        mean_exd[i, our_method_index] = -np.inf
        
        next_best_method_index = np.argmax(mean_exd[i, :])
        our_method_mean = mean[i, our_method_index]
        our_method_err = err[i, our_method_index]
        next_best_method_mean = mean[i, next_best_method_index]
        next_best_method_err = err[i, next_best_method_index]
        
        p_values = stats.ttest_ind_from_stats(our_method_mean, our_method_err, n_fold, \
            next_best_method_mean, next_best_method_err, n_fold)
        p_values = p_values.pvalue / 2
        stars = get_star_from_pvalue(p_values)
        star_y = max(our_method_mean, next_best_method_mean) + max(our_method_err, next_best_method_err) * 1.01
        star_x = i + width * (len(methods) - 0.5) * 1. / 2 - 0.1
        ax.text(star_x, star_y, stars, fontsize=LARGE_SIZE, ha='center')
        

def kaplan_meier_curve(ax, times, observations, labels, xlabel, ylabel, showCI=True,
                       max_time=None, min_y=0.2, usePercentageX=False, usePercentageY=True, colors=None, alpha=0.05,
                       pv_text_x=0.1,
                       pv_text_y=0.9,
                       show_y=True):
    kmf1 = KaplanMeierFitter()  # instantiate the class to create an object

    if type(colors) == str:
        colors = [colors for _ in range(len(times))]  # all the same color
    data = []
    for cohort in range(len(times)):
        dt = np.nan_to_num(times[cohort])
        for t, event in zip(dt, observations[cohort]):
            data.append([t, int(event), cohort])
    data = pd.DataFrame(data, columns=['duration', 'event', 'cohort'])
    cph = CoxPHFitter()
    cph.fit(data, duration_col='duration', event_col='event', formula="cohort")
    hazard_ratios = cph.summary["exp(coef)"]
    hr_text = f"H.R. = {hazard_ratios[0]:.2f}"
            
    for cohort in range(len(times)):
        dt = np.nan_to_num(times[cohort])
        kmf1.fit(dt, np.nan_to_num(observations[cohort]))
        kmf1.plot(ax=ax, label=labels[cohort], ci_show=showCI, color=colors[cohort] if colors is not None else None,
                  alpha=alpha, linewidth=2)
	
    p = logrank_test(times[0], times[1], observations[0], observations[1]).p_value
	# show p-value in this figure
    ax.text(pv_text_x, pv_text_y, r'$p$' + '= {:.1e}'.format(p), transform=ax.transAxes, \
	    	verticalalignment='top')
    print('p-value: {:.1e}'.format(p))

    if usePercentageX:
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(
            xmax=1, decimals=None, symbol='%', is_latex=False))
    if usePercentageY:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(
            xmax=1, decimals=None, symbol='%', is_latex=False))

    if max_time is not None:
        ax.set_xlim(0, max_time)
    ax.set_ylim(min_y, ax.get_ylim()[1])

    ax.set_xlabel(xlabel)
    if show_y:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    ax.text(pv_text_x, pv_text_y - 0.07, hr_text, transform=ax.transAxes, \
	    	verticalalignment='top')
    # disable legend
    ax.get_legend().remove()
    

def circular_bar_plot(ax, names, values, y_labels, hie_colors, rgrids=[]):
    '''
    The function is used to plot circular bar plot
	:param ax: axis
	:param names: names of the methods
	:param values: values of the methods
	:param y_labels: y labels
    '''
    width = 0.15
    bottom_start = 5
    srt_names = list(names)
    srt_values = list(values)
    angles = np.linspace(np.pi, -np.pi, len(values) + 3, endpoint=False)

    ct_angles = angles[0: len(srt_values)]
    ax.bar(ct_angles, np.asarray(srt_values), width=width, 
			color=hie_colors, 
            edgecolor='k', 
			bottom=bottom_start, zorder=3)
    for angle, label, val in zip(ct_angles, srt_names, srt_values):
        ax.text(angle, bottom_start + val + 0.3, label, 
			rotation=np.rad2deg(angle) + 180,
            fontsize=MEDIUM_SIZE,
			ha='center', va='center')
        # set the colors of that bar

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    rgrids = (np.asarray(rgrids))
    rgrids = [bottom_start + r for r in rgrids]
    ax.set_rgrids(rgrids, color='black')
    ax.set_thetagrids([], [])
    ax.grid(True, color=ax.spines['polar'].get_edgecolor(), linestyle='--', alpha=0.5)
    y_label_angle = np.pi * (len(srt_values) + 7) / (len(srt_values) + 3)
    for y_l in y_labels:
        ax.text(y_label_angle, bottom_start + np.log2(float(y_l[2:])) - 0.15, y_l,
                rotation=np.rad2deg(y_label_angle) - 90,
                fontsize=LARGE_SIZE,
                ha='center', va='center')
    ax.spines['polar'].set_visible(False)
    return srt_names, srt_values


def horizontal_bar_plot(ax, names, values, y_labels, hie_colors, n_cut=15, rgrids=[], show_percentage=False):
    '''
    The function is used to plot circular bar plot
	:param ax: axis
	:param names: names of the methods
	:param values: values of the methods
	:param y_labels: y labels
    '''
    height = 0.7
    srt_names = list(names)[:n_cut][::-1]
    srt_values = list(values)[:n_cut][::-1]
    srt_values = np.asarray(srt_values)
    hie_colors = hie_colors[:n_cut]

    ax.barh(srt_names, srt_values, height=height, 
			color=hie_colors)
    i = 0
    for label, val in zip(srt_names, srt_values):
        if show_percentage:
            text = '{:.2f}%'.format(100*val)
        else:
            text = int(val)
        ax.text(val + 0.85*val, i, text, 
            fontsize=MEDIUM_SIZE,
			ha='center', va='center')
        i += 1
        # set the colors of that bar
    #ax.set_xticks(rgrids)
    #ax.set_xticklabels(y_labels, fontsize=MEDIUM_SIZE)
    ax.set_xscale('log')
    ax.set_yticklabels(srt_names)
    plt.setp(ax.get_yticklabels(), rotation=-45, ha='right',
                rotation_mode="anchor")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return srt_names, srt_values


def mutation_6_tasks_barplot(axes, df_dict, plot_cols=['macro_auroc'], plot_tasks=[], plot_methods=[], y_names=[], bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1]):
    '''
    plot the bar plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    assert len(plot_tasks) <= axes.shape[0] * axes.shape[1]
    
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    n_rows, n_cols = axes.shape
    
    mid_point = len(plot_cols) / 2
    
    for i, t in enumerate(plot_tasks):
        i1, i2 = i // n_cols, i % n_cols
        for j, plot_col in enumerate(plot_cols):
            if j == 0:
                ax = axes[i1, i2]
            else:
                ax = axes[i1, i2].twinx()
            
            best_y, compare_col = -np.inf, ''
            for k, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][plot_col].to_list())
                y_std_err = np.std(df_dict[t][m][plot_col].to_list()) / \
                    np.sqrt(len(df_dict[t][m][plot_col].to_list()))
                ax.bar(j + (k + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar(j + (k + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=3)


                # remove the xticks
                ax.set_xticks([])
                if i2 == 0 and j == 0:
                    ax.set_ylabel(y_names[j])
                if i2 == n_cols - 1 and j == len(plot_cols) - 1:
                    # flip the y axis label text
                    #ax.yaxis.set_label_position("left")  # Set label position to left
                    ax.yaxis.label.set_rotation(-90)
                    ax.yaxis.label.set_verticalalignment('bottom')  # Adjust vertical alignment
                    ax.yaxis.label.set_horizontalalignment('right')
                    ax.set_ylabel(y_names[j])
                if y > best_y and k != 0:
                    best_y = y
                    compare_col = m
                    
            # add significance symbol
            delta_y = 0.01
            y_min = np.min([np.mean(df_dict[t][m][plot_col]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[t][m][plot_col]) + np.std(df_dict[t][m][plot_col].to_list()) / \
                    np.sqrt(len(df_dict[t][m][plot_col].to_list())) \
                        for m in plot_methods])
            y_h = df_dict[t][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[t][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = j + width
            x2 = j + (compare_idx + 1)*width
            
            if y_h > y_l and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            
            if j == 0:
                ax.axvspan(0, mid_point, facecolor=bg_colors[j], alpha=0.5, zorder=1)
            else:
                ax.axvspan(mid_point, len(plot_cols), facecolor=bg_colors[j], alpha=0.5, zorder=1)
            
            ax.set_ylim(floor_to_nearest(y_min), line_y + 1*delta_y)
            
            if i1 == 0:
                ax.title.set_text(t)
            if i1 == 1:
                ax.set_xlabel(t)
            
        plt.xlim(0, len(plot_cols))


def mutation_5_tasks_barplot(axes, df_dict, plot_col='macro_auroc', plot_tasks=[], plot_methods=[], y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1]):
    '''
    plot the bar plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
        # remove 'Prov170k-EGFR-Mutation'
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    plot_tasks.remove(f'Pan ' + r'$EFGR$' + ' subvariants')
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    for i, plot_task in enumerate(plot_tasks):
        ax = axes[i]
        
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            y = np.mean(df_dict[plot_task][m][plot_col])
            y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
            ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
            ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            if y > best_y and j != 0:
                best_y = y
                compare_col = m

        y_min = np.min([np.mean(df_dict[plot_task][m][plot_col]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[plot_task][m][plot_col]) + np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[plot_task][m][plot_col].to_list())) for m in plot_methods])
        y_h = df_dict[plot_task][plot_methods[0]][plot_col].to_list()
        y_l = df_dict[plot_task][compare_col][plot_col].to_list()
        
        p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        ax.set_xticks([])
        ax.set_xlabel(plot_task)
        if i == 0:
            ax.set_ylabel(y_name)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01

        stars = get_star_from_pvalue(p_value)
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width

        if y_h > y_l and len(stars) > 0:
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        format_ax(ax)
        ax.set_ylim(floor_to_nearest(y_min, 0.004), line_y + 1*delta_y)
        
        
def mutation_5_tasks_barplot_fixed_range(axes, df_dict, plot_col='macro_auroc', plot_tasks=[], plot_methods=[], y_name='AUROC', bg_colors=['#ffffcc', '#b3e2cd'], y_min=0, y_max=1, our_col=0, pop_col=-1, plot_xlabel=True):
    '''
    plot the bar plot for the mutation 6 tasks
    df_dict: results for the mutation 6 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
        # remove 'Prov170k-EGFR-Mutation'
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    plot_tasks.remove(f'Pan ' + r'$EFGR$' + ' subvariants')
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    
    for i, plot_task in enumerate(plot_tasks):
        ax = axes[i]
        
        dots_x, dots_y = [], []
        
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            y = np.mean(df_dict[plot_task][m][plot_col])
            y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
            
            # add the dots
            for y_i in df_dict[plot_task][m][plot_col].to_list():
                dots_x.append((j + 1)*width)
                dots_y.append(y_i)         
            
            ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
            ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            if y > best_y and j != our_col and j != pop_col:
                best_y = y
                compare_col = m
        
        # plot the dots
        add_bar_plot_dots(ax, dots_x, dots_y, s=5, jitter=width*0.6)

        y_h = df_dict[plot_task][plot_methods[our_col]][plot_col].to_list()
        y_l = df_dict[plot_task][compare_col][plot_col].to_list()
        
        p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        ax.set_xticks([])
        
        if plot_xlabel:
            ax.set_xlabel(plot_task)
            
        # get y tick label from the y_min and y_max, make sure only keep one decimal
        format_y_tick(ax, y_max, y_min)
            
        if i == 0:
            ax.set_ylabel(y_name)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.02

        stars = get_star_from_pvalue(p_value)
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = (our_col + 1)*width
        x2 = (compare_idx + 1)*width

        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        format_ax(ax)
        ax.set_ylim(y_min, y_max)
        

def mutation_5_gene_each_class_barplot(axes, df_dict, metric='AUROC', t='LUAD 5-Gene (TCGA)', bs=0.005):
    if metric == 'AUROC':
        col2groups = {'EGFR_auroc': r'$EGFR$', 'FAT1_auroc': r'$FAT1$', 'KRAS_auroc': r'$KRAS$', 'LRP1B_auroc': r'$LRP1B$', 'TP53_auroc': r'$TP53$'}
    else:
        col2groups = {'EGFR_auprc': r'$EGFR$', 'FAT1_auprc': r'$FAT1$', 'KRAS_auprc': r'$KRAS$', 'LRP1B_auprc': r'$LRP1B$', 'TP53_auprc': r'$TP53$'}
    
    plot_cols = list(col2groups.keys())
    plot_methods = list(df_dict[t].keys())
    
    n_groups = len(plot_cols)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    x = np.arange(n_groups)
    
    for i, plot_col in enumerate(plot_cols):
        ax = axes[i]
        
        best_y, compare_col = -np.inf, ''
        for j, m in enumerate(plot_methods):
            y = np.mean(df_dict[t][m][plot_col])
            y_std_err = np.std(df_dict[t][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[t][m][plot_col].to_list()))
            ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
            ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            if y > best_y and j != 0:
                best_y = y
                compare_col = m

        y_min = np.min([np.mean(df_dict[t][m][plot_col]) for m in plot_methods])
        y_max = np.max([np.mean(df_dict[t][m][plot_col]) + np.std(df_dict[t][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[t][m][plot_col].to_list())) for m in plot_methods])
        y_h = df_dict[t][plot_methods[0]][plot_col].to_list()
        y_l = df_dict[t][compare_col][plot_col].to_list()
        
        p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        ax.set_xticks([])
        ax.set_xlabel(col2groups[plot_col])
        if i == 0:
            ax.set_ylabel(metric)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.01
        stars = get_star_from_pvalue(p_value)
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = width
        x2 = (compare_idx + 1)*width

        if y_h > y_l and len(stars) > 0:
            ax.set_ylim(floor_to_nearest(y_min, bs), ceil_to_nearest(y_max, bs) + 2*delta_y)
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        else:
            ax.set_ylim(floor_to_nearest(y_min, bs), ceil_to_nearest(y_max, bs))
        format_ax(ax)


def mutation_5_gene_each_class_barplot_fixed_range(axes, df_dict, metric='AUROC', t='LUAD 5-Gene (TCGA)', bs=0.005, y_min=0.5, y_max=1, excld_col=[], our_meth_col=0):
    if metric == 'AUROC':
        col2groups = {'EGFR_auroc': r'$EGFR$', 'FAT1_auroc': r'$FAT1$', 'KRAS_auroc': r'$KRAS$', 'LRP1B_auroc': r'$LRP1B$', 'TP53_auroc': r'$TP53$'}
    else:
        col2groups = {'EGFR_auprc': r'$EGFR$', 'FAT1_auprc': r'$FAT1$', 'KRAS_auprc': r'$KRAS$', 'LRP1B_auprc': r'$LRP1B$', 'TP53_auprc': r'$TP53$'}
    
    plot_cols = list(col2groups.keys())
    plot_methods = list(df_dict[t].keys())
    
    n_groups = len(plot_cols)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    x = np.arange(n_groups)
    
    for i, plot_col in enumerate(plot_cols):
        ax = axes[i]
        
        best_y, compare_col = -np.inf, ''
        
        dots_x, dots_y = [], []
        for j, m in enumerate(plot_methods):
            y = np.mean(df_dict[t][m][plot_col])
            y_std_err = np.std(df_dict[t][m][plot_col].to_list()) / \
                        np.sqrt(len(df_dict[t][m][plot_col].to_list()))
            
            # add the dots
            for y_i in df_dict[t][m][plot_col].to_list():
                dots_x.append((j + 1)*width)
                dots_y.append(y_i)
                        
            ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
            ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            if y > best_y and j != our_meth_col and j not in excld_col:
                best_y = y
                compare_col = m
        
        # plot the dots
        add_bar_plot_dots(ax, dots_x, dots_y, s=5, jitter=width*0.6)
        
        y_h = df_dict[t][plot_methods[our_meth_col]][plot_col].to_list()
        y_l = df_dict[t][compare_col][plot_col].to_list()
        
        p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
        ax.set_xticks([])
        ax.set_xlabel(col2groups[plot_col])
        if i == 0:
            ax.set_ylabel(metric)
            
        format_y_tick(ax, y_max, y_min)
        #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # add significance symbol
        delta_y = 0.03
        stars = get_star_from_pvalue(p_value)
        compare_idx = plot_methods.index(compare_col)
        line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
        x1 = (our_meth_col + 1)*width
        x2 = (compare_idx + 1)*width

        if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
            ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
            ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
        
        ax.set_ylim(y_min, y_max)
        format_ax(ax)
        

def mutation_egfr_each_class_barplot(axes, df_dict, t='LUAD 5-Gene (TCGA)', bs=0.004):
    for m_i, metric in enumerate(['AUROC', 'AUPRC']):
        if metric == 'AUROC':
            col2groups = {'Exon 19 deletions_auroc': 'Exon 19\ndeletions', 'L858R_auroc': 'L858R', 'Others_auroc': 'Other subvariants'}
        else:
            col2groups = {'Exon 19 deletions_auprc': 'Exon 19\ndeletions', 'L858R_auprc': 'L858R', 'Others_auprc': 'Other subvariants'}
        
        plot_cols = list(col2groups.keys())
        plot_methods = list(df_dict[t].keys())
        
        n_groups = len(plot_cols)
        n_methods = len(plot_methods)
        width = 0.8 / n_methods # set the barplot width
        
        x = np.arange(n_groups)
        
        for i, plot_col in enumerate(plot_cols):
            ax = axes[m_i, i]
            
            best_y, compare_col = -np.inf, ''
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][plot_col])
                y_std_err = np.std(df_dict[t][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[t][m][plot_col].to_list()))
                ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=3)
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m

            y_min = np.min([np.mean(df_dict[t][m][plot_col]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[t][m][plot_col]) + np.std(df_dict[t][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[t][m][plot_col].to_list())) for m in plot_methods])
            y_h = df_dict[t][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[t][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            ax.set_ylim(floor_to_nearest(y_min, bs), ceil_to_nearest(y_max, bs))
            ax.set_xticks([])
            ax.set_xlabel(col2groups[plot_col])
            ax.set_ylabel(metric)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # add significance symbol
            delta_y = 0.01

            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = width
            x2 = (compare_idx + 1)*width

            if y_h > y_l and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            format_ax(ax)
        

def mutation_18_biomarker_each_class_plot(ax, df_dict, metric='AUROC', min_y=0.55, max_y=0.8, rotation=30,plot_title=True):
    t = 'Pan 18-biomarker'
    if metric == 'AUROC':
        col2groups = {'PD-L1_auroc': r'$PD$-$L1$', 'TP53_auroc': r'$TP53$', 'LRP1B_auroc': r'$LRP1B$', 'KRAS_auroc': r'$KRAS$', 'APC_auroc': r'$APC$', 'KMT2D_auroc': r'$KMT2D$', 'FAT1_auroc': r'$FAT1$', 'SPTA1_auroc': r'$SPTA1$', 'ZFHX3_auroc': r'$ZFHX3$', 'KMT2C_auroc': r'$KMT2C$', 'EGFR_auroc': r'$EGFR$', 'ARID1A_auroc': r'$ARID1A$', 'PIK3CA_auroc': r'PIK3CA', 'PRKDC_auroc': r'$PRKDC$', 'NOTCH1_auroc': r'$NOTCH1$', 'ATM_auroc': r'$ATM$', 'KMT2A_auroc': r'$KMT2A$', 'ROS1_auroc': r'$ROS1$'}
    else:
        col2groups = {'PD-L1_auprc': r'$PD$-$L1$', 'TP53_auprc': r'$TP53$', 'LRP1B_auprc': r'$LRP1B$', 'KRAS_auprc': r'$KRAS$', 'APC_auprc': r'$APC$', 'KMT2D_auprc': r'$KMT2D$', 'FAT1_auprc': r'$FAT1$', 'SPTA1_auprc': r'$SPTA1$', 'ZFHX3_auprc': r'$ZFHX3$', 'KMT2C_auprc': r'$KMT2C$', 'EGFR_auprc': r'$EGFR$', 'ARID1A_auprc': r'$ARID1A$', 'PIK3CA_auprc': r'PIK3CA', 'PRKDC_auprc': r'$PRKDC$', 'NOTCH1_auprc': r'$NOTCH1$', 'ATM_auprc': r'$ATM$', 'KMT2A_auprc': r'$KMT2A$', 'ROS1_auprc': r'$ROS1$'}
    
    plot_cols = list(col2groups.keys())
    plot_methods = list(df_dict[t].keys())
    y = [np.mean(df_dict[t][plot_methods[0]][plot_col]) for plot_col in plot_cols]
    srt_idx = np.argsort(np.asarray(y))[::-1]
    plot_cols = [plot_cols[i] for i in srt_idx]
    plot_methods = np.asarray(plot_methods)
    
    n_groups = len(plot_cols)
    
    x = np.arange(n_groups)
    
    for i, m in enumerate(plot_methods):
        y = [np.mean(df_dict[t][m][plot_col]) for plot_col in plot_cols]
        if m == 'Prov-MSR-Path' or m == 'Prov-GigaPath':
            zorder = 3
        else:
            zorder = 2
        ax.scatter(x, y, s=20, label=m, color=COLORS[m], zorder=zorder, marker=MARKERS[m])
    format_ax(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([col2groups[plot_col] for plot_col in plot_cols], rotation=rotation, ha='right')
    ax.set_ylabel(metric)
    ax.set_ylim(min_y, max_y)
    ax.legend(loc='upper right', frameon=False)
    
    if plot_title:
        ax.set_title(t)
    

def subtyping_9_cancers_barplot_half_half(axes, df_dict, plot_cols=['test_auc', 'test_acc'], plot_tasks=[], plot_methods=[], y_names=['AUROC', 'Accuracy'], bg_colors=['#ffffcc', '#b3e2cd'], y_max=[-1, -1, -1, -1, -1, -1]):
    '''
    plot the bar plot for the cancer subtyping 9 tasks
    df_dict: results for the cancer subtyping 9 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    assert len(plot_tasks) <= axes.shape[0] * axes.shape[1]
    
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    n_rows, n_cols = axes.shape
    
    mid_point = len(plot_cols) / 2
    
    for i, t in enumerate(plot_tasks):
        i1, i2 = i // n_cols, i % n_cols
        for j, plot_col in enumerate(plot_cols):
            if j == 0:
                ax = axes[i1, i2]
            else:
                ax = axes[i1, i2].twinx()
            
            best_y, compare_col = -np.inf, ''
            for k, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][plot_col].to_list())
                y_std_err = np.std(df_dict[t][m][plot_col].to_list()) / \
                    np.sqrt(len(df_dict[t][m][plot_col].to_list()))
                ax.bar(j + (k + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar(j + (k + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=3)


                # remove the xticks
                ax.set_xticks([])
                if i2 == 0 and j == 0:
                    ax.set_ylabel(y_names[j])
                if i2 == n_cols - 1 and j == len(plot_cols) - 1:
                    # flip the y axis label text
                    #ax.yaxis.set_label_position("left")  # Set label position to left
                    ax.yaxis.label.set_rotation(-90)
                    ax.yaxis.label.set_verticalalignment('bottom')  # Adjust vertical alignment
                    ax.yaxis.label.set_horizontalalignment('right')
                    ax.set_ylabel(y_names[j])
                if y > best_y and k != 0:
                    best_y = y
                    compare_col = m
                    
            # add significance symbol
            delta_y = 0.01
            y_min = np.min([np.mean(df_dict[t][m][plot_col]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[t][m][plot_col]) + np.std(df_dict[t][m][plot_col].to_list()) / \
                    np.sqrt(len(df_dict[t][m][plot_col].to_list())) \
                        for m in plot_methods])
            y_h = df_dict[t][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[t][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = j + width
            x2 = j + (compare_idx + 1)*width
            
            if y_h > y_l and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            
            if j == 0:
                ax.axvspan(0, mid_point, facecolor=bg_colors[j], alpha=0.5, zorder=1)
            else:
                ax.axvspan(mid_point, len(plot_cols), facecolor=bg_colors[j], alpha=0.5, zorder=1)
            
            if len(stars) > 0:
                ax.set_ylim(floor_to_nearest(y_min), line_y + 3.5*delta_y)
            else:
                ax.set_ylim(floor_to_nearest(y_min), line_y + 1*delta_y)
            ax.set_xlabel(t)
            
        plt.xlim(0, len(plot_cols))
        

def subtyping_9_cancers_barplot(axes, df_dict, plot_cols=['test_auc', 'test_acc'], plot_tasks=[], plot_methods=[], y_names=['AUROC', 'Accuracy'], y_max=[-1, -1, -1, -1, -1, -1]):
    '''
    plot the bar plot for the cancer subtyping 9 tasks
    df_dict: results for the cancer subtyping 9 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    ax_rows, ax_cols = axes.shape
    for plot_col in plot_cols:
        ax_n = 0
        for i, plot_task in enumerate(plot_tasks):
            ax_i, ax_j = ax_n // int(ax_cols / 2), ax_n % int(ax_cols / 2) + int(ax_cols / 2) * plot_cols.index(plot_col)
            ax = axes[ax_i, ax_j]
            ax_n += 1
            
            best_y, compare_col = -np.inf, ''
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[plot_task][m][plot_col])
                y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
                ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m

            y_min = np.min([np.mean(df_dict[plot_task][m][plot_col]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[plot_task][m][plot_col]) + np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list())) for m in plot_methods])
            y_h = df_dict[plot_task][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[plot_task][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            ax.set_xticks([])
            ax.set_xlabel(plot_task)
            y_name = y_names[plot_cols.index(plot_col)]
            if ax_j == 0 or ax_j == int(ax_cols/2):
                ax.set_ylabel(y_name)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # add significance symbol
            delta_y = 0.01

            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = width
            x2 = (compare_idx + 1)*width

            if y_h > y_l and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            format_ax(ax)
            ax.set_ylim(floor_to_nearest(y_min, 0.004) - 3*delta_y, line_y + 1*delta_y)
            

def subtyping_9_cancers_barplot_fixed_range(axes, df_dict, plot_cols=['test_auc', 'test_acc'], plot_tasks=[], plot_methods=[], y_names=['AUROC', 'Accuracy'], y_min=0.5, y_max=1):
    '''
    plot the bar plot for the cancer subtyping 9 tasks
    df_dict: results for the cancer subtyping 9 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    ax_rows, ax_cols = axes.shape
    for plot_col in plot_cols:
        ax_n = 0
        
        for i, plot_task in enumerate(plot_tasks):
            ax_i, ax_j = ax_n // int(ax_cols / 2), ax_n % int(ax_cols / 2) + int(ax_cols / 2) * plot_cols.index(plot_col)
            ax = axes[ax_i, ax_j]
            ax_n += 1
            
            dots_x, dots_y = [], []
            
            best_y, compare_col = -np.inf, ''
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[plot_task][m][plot_col])
                y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
                            
                # add the dots
                for y_i in df_dict[plot_task][m][plot_col].to_list():
                    dots_x.append((j + 1)*width)
                    dots_y.append(y_i)
                    
                ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m
            
            # plot the dots
            add_bar_plot_dots(ax, dots_x, dots_y, s=5, jitter=width*0.6)

            y_h = df_dict[plot_task][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[plot_task][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            ax.set_xticks([])
            ax.set_xlabel(plot_task)
            y_name = y_names[plot_cols.index(plot_col)]
            if ax_j == 0 or ax_j == int(ax_cols/2):
                ax.set_ylabel(y_name)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # add significance symbol
            delta_y = 0.02

            stars = get_star_from_pvalue(p_value)
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
            ax.set_ylim(y_min, y_max)


def subtyping_9_cancers_ablation_barplot(axes, df_dict, plot_cols=['test_auc', 'test_acc'], plot_tasks=[], plot_methods=[], y_names=['AUROC', 'Accuracy'], y_max=[-1, -1, -1, -1, -1, -1]):
    '''
    plot the bar plot for the cancer subtyping 9 tasks
    df_dict: results for the cancer subtyping 9 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    ax_rows, ax_cols = axes.shape
    for plot_col in plot_cols:
        ax_n = 0
        for i, plot_task in enumerate(plot_tasks):
            ax_i, ax_j = ax_n // int(ax_cols / 2), ax_n % int(ax_cols / 2) + int(ax_cols / 2) * plot_cols.index(plot_col)
            ax = axes[ax_i, ax_j]
            ax_n += 1
            
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[plot_task][m][plot_col])
                y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
                ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            
            compare_col = 'Prov-MSR-Path w. ABMIL'

            y_min = np.min([np.mean(df_dict[plot_task][m][plot_col]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[plot_task][m][plot_col]) + np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list())) for m in plot_methods])
            y_h = df_dict[plot_task][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[plot_task][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            ax.set_xticks([])
            ax.set_xlabel(plot_task)
            y_name = y_names[plot_cols.index(plot_col)]
            if ax_j == 0 or ax_j == int(ax_cols/2):
                ax.set_ylabel(y_name)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # add significance symbol
            delta_y = 0.01

            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = width
            x2 = (compare_idx + 1)*width

            if y_h > y_l and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            format_ax(ax)
            ax.set_ylim(floor_to_nearest(y_min, 0.004) - 3*delta_y, line_y + 1*delta_y)


def subtyping_9_cancers_ablation_barplot_fixed_range(axes, df_dict, plot_cols=['test_auc', 'test_acc'], plot_tasks=[], plot_methods=[], y_names=['AUROC', 'Accuracy'], y_min=0.5, y_max=1):
    '''
    plot the bar plot for the cancer subtyping 9 tasks
    df_dict: results for the cancer subtyping 9 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width
    
    ax_rows, ax_cols = axes.shape
    for plot_col in plot_cols:
        ax_n = 0
        for i, plot_task in enumerate(plot_tasks):
            ax_i, ax_j = ax_n // int(ax_cols / 2), ax_n % int(ax_cols / 2) + int(ax_cols / 2) * plot_cols.index(plot_col)
            ax = axes[ax_i, ax_j]
            ax_n += 1
            
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[plot_task][m][plot_col])
                y_std_err = np.std(df_dict[plot_task][m][plot_col].to_list()) / \
                            np.sqrt(len(df_dict[plot_task][m][plot_col].to_list()))
                ax.bar((j + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((j + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=4)
            
            compare_col = 'Prov-MSR-Path w. ABMIL'

            y_h = df_dict[plot_task][plot_methods[0]][plot_col].to_list()
            y_l = df_dict[plot_task][compare_col][plot_col].to_list()
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            ax.set_xticks([])
            ax.set_xlabel(plot_task)
            y_name = y_names[plot_cols.index(plot_col)]
            if ax_j == 0 or ax_j == int(ax_cols/2):
                ax.set_ylabel(y_name)
            #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            
            # add significance symbol
            delta_y = 0.02

            stars = get_star_from_pvalue(p_value)
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
            ax.set_ylim(y_min, y_max)


def vlp_subtyping_barplot(axes, df_dict, plot_tasks=[], plot_methods=[], y_names=[]):
    '''
    plot the bar plot for the vlp cancer subtyping 3 tasks
    df_dict: results for the vlp cancer subtyping 3 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    if len(y_names) == 0:
        y_names = list(df_dict[plot_tasks[0]][plot_methods[0]].keys())

    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width

    x = np.arange(n_groups)

    for a_i, y_name in enumerate(y_names):
        ax = axes[a_i]
        for i, m in enumerate(plot_methods):
            y = [np.mean(df_dict[t][m][y_name]) for t in plot_tasks]
            y_std_err = [np.std(df_dict[t][m][y_name]) / \
                        np.sqrt(len(df_dict[t][m][y_name])) for t in plot_tasks]
            ax.bar(x + (i + 1)*width, y, width, label=m, color=COLORS[m])
            ax.errorbar(x + (i + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2)
        
        best_y, compare_col = -np.inf, ''
        y_max_tasks, y_min_tasks = 0, 1
        
        # add significance symbol
        for i, t in enumerate(plot_tasks):
            best_y, compare_col = -np.inf, ''
            dots_x, dots_y = [], []
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][y_name])
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m
                
                # add the dots
                for y_i in df_dict[t][m][y_name]:
                    dots_x.append(i + (j + 1)*width)
                    dots_y.append(y_i)
            
            # plot the dots
            # add_bar_plot_dots(ax, dots_x, dots_y, s=5, jitter=width*0.6)
                    
            delta_y = 0.01
            y_min = np.min([np.mean(df_dict[t][m][y_name]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[t][m][y_name]) + np.std(df_dict[t][m][y_name]) / \
                    np.sqrt(len(df_dict[t][m][y_name])) for m in plot_methods])
            if y_min < y_min_tasks:
                y_min_tasks = y_min
            if y_max > y_max_tasks:
                y_max_tasks = y_max

            y_h = df_dict[t][plot_methods[0]][y_name]
            y_l = df_dict[t][compare_col][y_name]

            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = i + width
            x2 = i + (compare_idx + 1)*width

            if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')

        y_min = floor_to_nearest(y_min_tasks, 0.05)
        y_max = ceil_to_nearest(y_max_tasks, 0.05)
        ax.set_ylim(y_min, y_max)
        format_ax(ax)
        ax.set_xticks(x + width * ((n_methods) / 2 + 1))
        ax.set_xticklabels(plot_tasks)
        ax.set_ylabel(y_name)
        #ax.legend(loc='upper right', frameon=False)
        

def vlp_muts_barplot(axes, df_dict, plot_tasks=[], plot_methods=[], y_names=[]):
    '''
    plot the bar plot for the vlp cancer subtyping 3 tasks
    df_dict: results for the vlp cancer subtyping 3 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    if len(y_names) == 0:
        y_names = list(df_dict[plot_tasks[0]][plot_methods[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width

    x = np.arange(n_groups)
    
    #n_rows, n_cols = axes.shape
    
    for a_i, y_name in enumerate(y_names):
        for a_j, t in enumerate(plot_tasks):
            if len(y_names) == 1:
                ax = axes[a_j]
            else:
                ax = axes[a_i, a_j]
            for i, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][y_name])
                y_std_err = np.std(df_dict[t][m][y_name]) / \
                            np.sqrt(len(df_dict[t][m][y_name]))
                ax.bar((i + 1)*width, y, width, label=m, color=COLORS[m], zorder=3)
                ax.errorbar((i + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2, zorder=3)
        
            best_y, compare_col = -np.inf, ''
            
            best_y, compare_col = -np.inf, ''
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][y_name])
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m
            delta_y = 0.01
            y_min = np.min([np.mean(df_dict[t][m][y_name]) for m in plot_methods])
            y_max = np.max([np.mean(df_dict[t][m][y_name]) + np.std(df_dict[t][m][y_name]) / \
                    np.sqrt(len(df_dict[t][m][y_name])) for m in plot_methods])
                
            y_h = df_dict[t][plot_methods[0]][y_name]
            y_l = df_dict[t][compare_col][y_name]
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = width
            x2 = (compare_idx + 1)*width
            
            if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
            
            y_min = floor_to_nearest(y_min, 0.05)
            y_max = ceil_to_nearest(y_max + 0.01)
            ax.set_ylim(y_min, y_max)
            format_ax(ax)
            ax.set_xticks([])
            if a_i == 1 or len(y_names) == 1:
                # italic the task name
                ax.set_xlabel(t, fontstyle='italic')
            ax.set_ylabel(y_name)
            #if a_n == 0:
            #    ax.legend(loc='upper center', frameon=False, ncol=len(plot_methods))
        #ax.legend(loc='upper right', frameon=False)
        
        
def vlp_muts_barplot_fixed_range(axes, df_dict, plot_tasks=[], plot_methods=[], y_names=[], y_min=0.5, y_max=1):
    '''
    plot the bar plot for the vlp cancer subtyping 3 tasks
    df_dict: results for the vlp cancer subtyping 3 tasks and all comparison approaches
    '''
    if len(plot_tasks) == 0:
        plot_tasks = list(df_dict.keys())
    if len(plot_methods) == 0:
        plot_methods = list(df_dict[plot_tasks[0]].keys())
    if len(y_names) == 0:
        y_names = list(df_dict[plot_tasks[0]][plot_methods[0]].keys())
    
    n_groups = len(plot_tasks)
    n_methods = len(plot_methods)
    width = 0.8 / n_methods # set the barplot width

    x = np.arange(n_groups)
    
    #n_rows, n_cols = axes.shape
    
    for a_i, y_name in enumerate(y_names):
        for a_j, t in enumerate(plot_tasks):
            if len(y_names) == 1:
                ax = axes[a_j]
            else:
                ax = axes[a_i, a_j]
            for i, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][y_name])
                y_std_err = np.std(df_dict[t][m][y_name]) / \
                            np.sqrt(len(df_dict[t][m][y_name]))
                ax.bar((i + 1)*width, y, width, label=m, color=COLORS[m])
                ax.errorbar((i + 1)*width, y, yerr=y_std_err, fmt='none', ecolor='k', capsize=2)
        
            best_y, compare_col = -np.inf, ''
            
            best_y, compare_col = -np.inf, ''
            for j, m in enumerate(plot_methods):
                y = np.mean(df_dict[t][m][y_name])
                if y > best_y and j != 0:
                    best_y = y
                    compare_col = m
            delta_y = 0.01
            
            y_h = df_dict[t][plot_methods[0]][y_name]
            y_l = df_dict[t][compare_col][y_name]
            
            p_value = wilcoxon(y_h, y_l, alternative='greater').pvalue
            stars = get_star_from_pvalue(p_value)
            compare_idx = plot_methods.index(compare_col)
            line_y = np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + delta_y
            x1 = width
            x2 = (compare_idx + 1)*width
            
            if np.mean(y_h) > np.mean(y_l) and len(stars) > 0:
                ax.plot([x1, x1], [np.mean(y_h) + np.std(y_h)/np.sqrt(len(y_h)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x2, x2], [np.mean(y_l) + np.std(y_l)/np.sqrt(len(y_l)) + 0.5*delta_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
                ax.text((x1 + x2)/2, line_y, stars, fontsize=7, ha='center', va='bottom')
                
            ax.set_ylim(y_min, y_max)
            format_ax(ax)
            ax.set_xticks([])
            if a_i == 1 or len(y_names) == 1:
                # italic the task name
                ax.set_xlabel(t, fontstyle='italic')
            ax.set_ylabel(y_name)
            #if a_n == 0:
            #    ax.legend(loc='upper center', frameon=False, ncol=len(plot_methods))
        #ax.legend(loc='upper right', frameon=False)

    
def vlp_scatter_plot(ax, df_dict, cmp_m='MI-Zero', task='NSCLC', metric='BACC', show_y=True, s=6, show_legend=False, title=''):
    '''
    plot the scatter plot for the VLP cancer subtyping tasks.
    df_dict: results for the VLP cancer subtyping tasks and all comparison approaches
    '''

    # Extract data
    y_h = df_dict[task]['Prov-GigaPath'][metric]
    y_l = df_dict[task][cmp_m][metric]

    # Determine which data points are higher
    idx_h = y_h > y_l

    # Plot scatter points
    ax.scatter(y_l[idx_h], y_h[idx_h], s=s, c=COLORS['Prov-MSR-Path'], marker=MARKERS['Prov-MSR-Path'], label='Prov-MSR-Path > {}'.format(cmp_m), alpha=1)
    ax.scatter(y_l[~idx_h], y_h[~idx_h], s=s, c=COLORS[cmp_m], marker=MARKERS[cmp_m], label='Prov-MSR-Path < {}'.format(cmp_m))

    # Find overall min and max
    v_min, v_max = np.min([np.min(y_h), np.min(y_l)]), np.max([np.max(y_h), np.max(y_l)])

    # Adjust to nearest value if needed
    v_min = floor_to_nearest(v_min, 0.01) - 0.005
    v_max = ceil_to_nearest(v_max, 0.01) + 0.01

    # Set identical limits for x and y axis
    ax.set_xlim(v_min, v_max)
    ax.set_ylim(v_min, v_max)

    # Plot diagonal line
    ax.plot([v_min, v_max], [v_min, v_max], c='#636363', linewidth=1)

    # Format axis
    format_ax(ax)

    # Set labels
    ax.set_xlabel(cmp_m)
    if show_y:
        ax.set_ylabel('Prov-MSR-Path')

    # Add legend
    if show_legend:
        ax.legend(loc='lower right', frameon=False)

    # Set title
    if 'subtyping' in title:
        ax.set_title(title.split(' ')[0])
    else:
        ax.set_title(title, fontstyle='italic')

    # Sync tick labels for both axes
    # tk = ax.get_xticks()
    ticks= [0.46, 0.50, 0.54, 0.58, 0.62, 0.66]
    ticks_labels = ['0.46', '0.50', '0.54', '0.58', '0.62', '0.66']
    ticks = [t for t in ticks if t >= v_min and t <= v_max]
    ticks_labels = [t for t in ticks_labels if eval(t) >= v_min and eval(t) <= v_max]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks_labels)
    #ax.set_xticks(ax.get_xticks())
    #ax.set_xticklabels(ax.get_xticklabels())
                

def add_significance_symbol_boxplot(ax, merged_df, our_method_name='Prov-MSR-Path', compare_col='AUROC', baseline=r'HIPT-$Prov$-$Path$'):
    merged_df = merged_df.copy()
    merged_df = pd.DataFrame(merged_df)
    methods = merged_df['methods'].unique().tolist()
    tasks = merged_df['task'].unique().tolist()
    for t in tasks:
        our_score = merged_df.loc[(merged_df['methods'] == our_method_name) & (merged_df['task'] == t)][compare_col].to_list()
        if len(baseline) > 0 and baseline in methods:
            best_score = merged_df.loc[(merged_df['methods'] == baseline) & (merged_df['task'] == t)][compare_col].to_list()
            best_idx = methods.index(baseline)
        else:
            best_score, best_idx = -np.inf, -1
            for m in methods:
                if m == our_method_name:
                    continue
                score = merged_df.loc[(merged_df['methods'] == m) & (merged_df['task'] == t)][compare_col].to_list()
                if np.median(score) > np.median(best_score):
                    best_score = score
                    best_idx = methods.index(m)
        if np.median(our_score) < np.median(best_score):
            continue
        p_value = wilcoxon(our_score, best_score, alternative='greater').pvalue
        stars = get_star_from_pvalue(p_value)
        if len(stars) == 0:
            continue
        dt = 0.05
        delta_y = dt
        our_method_index = methods.index(our_method_name)
        line_y = max(our_score) + delta_y
        x1 = tasks.index(t) + 0.8/(len(methods)) * (our_method_index - len(methods)/2 + 0.5)
        x2 = tasks.index(t) + 0.8/(len(methods)) * (best_idx - len(methods)/2 + 0.5)
        
        ax.plot([x1, x1], [line_y - 0.4*dt, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        ax.plot([x2, x2], [max(best_score) + 0.5*dt, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        ax.plot([x1, x2], [line_y, line_y], c=ax.spines['bottom'].get_edgecolor(), linewidth=1)
        ax.text((x1 + x2)/2, line_y - 0.2*dt, stars, fontsize=LARGE_SIZE, ha='center')