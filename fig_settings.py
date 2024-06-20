import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.font_manager

# set basic parameters
mpl.rcParams['pdf.fonttype'] = 42

is_black_background = False
if is_black_background:
    plt.style.use('dark_background')
    mpl.rcParams.update({"ytick.color" : "w",
                     "xtick.color" : "w",
                     "axes.labelcolor" : "w",
                     "axes.edgecolor" : "w"})

MEDIUM_SIZE = 8
SMALLER_SIZE = 6
plt.rc('font', size=MEDIUM_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('axes', titlesize=MEDIUM_SIZE)	 # fontsize of the axes title
plt.rc('xtick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALLER_SIZE)	 # fontsize of the tick labels
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rc('legend', fontsize=SMALLER_SIZE)
# plt.rc('font', family='Helvetica')
mpl.rcParams.update({
    "pdf.use14corefonts": True
})
 #, xtick.color='w', axes.labelcolor='w', axes.edge_color='w'
FIG_HEIGHT = 18 / 2.54
FIG_WIDTH = 18 / 2.54


ABL_COLORS = {
    #'LongNet w. BERT': '#01665e',
    #'LongNet w. MAE/mse': '#35978f',
    'GigaPath': '#121060',
    'Prov-MSR-Path': '#121060',
    'GigaPath fz': '#beaed4',
    'GigaPath w/o pt': '#fdc086',
    'GigaPath w. transformer': '#ffff99',
    'GigaPath w. ABMIL': '#386cb0',
    'GigaPath (Dinov2) enc w. ABMIL': '#f0027f',
}

COLORS_AIREADI_compute = {
    "MAE-joint finetune": "#0E3E87",
    "MAE-joint lock_part": "#80BFC9",
    "retfound finetune": "#346CAC",
    "retfound lock_part": "#D9EAEA",
    "MAE-joint linear_probe": "#F7E474",
    "retfound linear_probe": "#D8B258",

    # "MAE2D 3D": "#DEEAEA", 
}
# Original Blue (#0E3E87) - Deep blue.
# Modified Blue (#1A5298) - Slightly lighter and shifted towards teal.
# Original Light Blue (#346CAC) - Light blue.
# Modified Light Blue (#4981BF) - A bit lighter and less saturated.
# Original Yellow (#F7E474) - Bright yellow.
# Modified Yellow (#F8EC8C
COLORS_AIREADI = {
    "MAE-joint": "#0E3E87",
    "MAE-joint 3D": "#0E3E87",
    "retFound 3D": "#346CAC",
    "retfound 3D": "#346CAC",
    "retFound 2D": "#F7E474",
    "retfound 2D": "#F7E474",
    # "retFound 2D": "#DEEAEA",
    # "retfound 2D": "#DEEAEA",
    # "from_scratch 3D": "#F7E474",
    # "from_scratch 2D": "#D8B258",
    "MAE2D 3D": "#DEEAEA", 
}

COLORS = {
    "MAE-joint": "#0E3E87",
    "MAE-joint 3D": "#0E3E87",
    "retFound 3D": "#346CAC",
    "retfound 3D": "#346CAC",

    "retFound 2D": "#DEEAEA",
    "retfound 2D": "#DEEAEA",
    "from_scratch 3D": "#F7E474",
    "from_scratch 2D": "#D8B258",
    "MAE2D 3D": "#F7E474",
 
 
    # 'MAE-joint': '#121060',
    # 'MAE-joint 3D': '#121060',
    # 'retFound 3D': '#9C9CBD',
    # 'retfound 3D': '#9C9CBD',
    # 'retFound 2D': '#C8A4B6',
    # 'retfound 2D': '#C8A4B6',
    # 'from_scratch 3D': '#F1DEBB',
    # 'from_scratch 2D': '#9C9CBD',

    #'LongNet w. BERT': '#01665e',
    #'LongNet w. MAE/mse': '#35978f',
    'GigaPath': '#121060',
    'GigaPath-TCGA': '#91bfdb',
    'Prov-MSR-Path': '#121060',
    'Prov-GigaPath': '#121060',
    'Prov-MSR-Path (inductive)': '#121060',
    'Prov-MSR-Path (transductive)': '#636363',
    'HIPT-TCGA': '#9C9CBD',
    'HIPT': '#9C9CBD',
    r'HIPT-$Prov$-$Path$': '#fdc086',
    'CtransPath': '#C8A4B6',
    'REMEDIS': '#F1DEBB',
    #'SSL-Dinov2': '#121060',
    'GigaPath (Small)': '#d1e5f0',
    'SSL-Dinov2': '#121060',
    'SSL-MAE': '#9C9CBD',
    'SSL-SimCLR': '#C8A4B6',
    'SL-ImageNet': '#F1DEBB',
    'MI-Zero': '#9C9CBD',
    'BiomedCLIP': '#C8A4B6',
    'PLIP': '#F1DEBB',
    'Prov-MSR-Path fz': '#9C9CBD',
    'Prov-MSR-Path w. transformer': '#fdc086',
    'Prov-MSR-Path w/o pt': '#C8A4B6',
    'Prov-MSR-Path w. ABMIL': '#386cb0',
    'Prov-GigaPath fz': '#9C9CBD',
    'Prov-GigaPath w. transformer': '#fdc086',
    'Prov-GigaPath w/o pt': '#C8A4B6',
    'Prov-GigaPath w. ABMIL': '#386cb0',
}

MARKERS = {
    'GigaPath': 'o',
    'Prov-MSR-Path': 'o',
    'Prov-GigaPath': 'o',
    'HIPT-TCGA': 'x',
    'HIPT': 'x',
    r'HIPT-$Prov$-$Path$': 'P',
    'CtransPath': 's',
    'REMEDIS': '*',
    'MI-Zero': 'x',
    'BiomedCLIP': 'x',

}

def get_mutation_axis():
    fig = plt.figure(figsize=(4*FIG_WIDTH, 3.5*FIG_HEIGHT))
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[1, 1])
    return fig, gs

def get_survival_axis():
    fig = plt.figure(figsize=(5*FIG_WIDTH, 3.5*FIG_HEIGHT))
    gs = gridspec.GridSpec(nrows=3, ncols=20, height_ratios=[1.5, 1, 1])
    return fig, gs

def get_survival_axis1():
    fig, ax = plt.subplots(figsize=(3.5*FIG_WIDTH, 1.1*FIG_HEIGHT))
    return fig, ax

def get_survival_axis2():
    fig, axes = plt.subplots(figsize=(3.5*FIG_WIDTH, 0.9*FIG_HEIGHT), ncols=5)
    return fig, axes

def get_survival_axis3():
    fig, axes = plt.subplots(figsize=(3.5*FIG_WIDTH, 0.9*FIG_HEIGHT), ncols=4)
    return fig, axes

def get_radar_axis():
    fig, ax = plt.subplots(figsize=(2.9*FIG_WIDTH, 3.1*FIG_HEIGHT), subplot_kw={'projection': 'polar'})
    return fig, ax

def get_vertical_axis():
    fig, ax = plt.subplots(figsize=(1.5*FIG_WIDTH, 1.6*FIG_HEIGHT))
    return fig, ax

def get_subtyping_axis():
    fig, ax = plt.subplots(figsize=(2.5*FIG_WIDTH, 1.0*FIG_HEIGHT))
    return fig, ax

def get_ablation_axis():
    fig, ax = plt.subplots(figsize=(4*FIG_WIDTH, 1.25*FIG_HEIGHT), ncols=2)
    return fig, ax

def get_zeroshot_axis():
    fig, ax = plt.subplots(figsize=(1.15*FIG_WIDTH, 1.15*FIG_HEIGHT))
    return fig, ax

def get_circular_axis():
    fig, ax = plt.subplots(figsize=(5*FIG_WIDTH, 2.7*FIG_HEIGHT), subplot_kw={'projection': 'polar'}, ncols=2)
    return fig, ax

def get_box_plot_axis():
    fig, ax = plt.subplots(figsize=(1.5*FIG_WIDTH, 1.5*FIG_HEIGHT))
    return fig, ax

def get_wider_axis():
    fig, ax = plt.subplots(figsize=(3*FIG_WIDTH, 1.5*FIG_HEIGHT))
    return fig, ax

def get_umap_axis():
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 2*FIG_HEIGHT))
    return fig, ax

def get_km_plot_axis():
    fig, ax = plt.subplots(figsize=(5*FIG_WIDTH, 1.25*FIG_HEIGHT), ncols=5)
    return fig, ax

def get_km_late_stage_plot_axis():
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 1.25*FIG_HEIGHT), ncols=2)
    return fig, ax

def get_surv_stage_plot_axis():
    fig, ax = plt.subplots(figsize=(2*FIG_WIDTH, 1.25*FIG_HEIGHT), ncols=2)
    return fig, ax

def get_box_plot_setting():
    return {
        'box alpha': 1,
        'box linewidth': 1.5,
        'cap linewidth': 1.5,
        'whisker linestyle': '--',
        'whisker linewidth': 1.5,
        'median color': 'k',
        'median linewidth': 1.5,
        'marker size': 15,
        'marker edge color': 'k',
    }
    
def get_bar_plot_setting():
    return {
        'capsize': 5,
        'capthick': 1.5,
    }