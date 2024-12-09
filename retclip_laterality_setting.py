
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *

from scipy.stats import ttest_rel, ttest_ind

def get_results_df(octcube_df, retfound_3d_df, retfound_2d_df, octcube_ir_df, retfound_3d_ir_df, retfound_2d_ir_df):
    results_dict = {}
    results_dict['MAE-joint'] = {}
    precision_prefix = 'Precision@'
    oct2ir_prefix = 'OCT_to_IR'
    ir2oct_prefix = 'IR_to_OCT'
    for idx, row in octcube_df.iterrows():
        for col in octcube_df.columns:
            results_dict['MAE-joint'][f'{oct2ir_prefix} {precision_prefix}{col}'] = row[col]
    for idx, row in octcube_ir_df.iterrows():
        for col in octcube_ir_df.columns:
            results_dict['MAE-joint'][f'{ir2oct_prefix} {precision_prefix}{col}'] = row[col]
    results_dict['retFound 3D'] = {}
    for idx, row in retfound_3d_df.iterrows():
        for col in retfound_3d_df.columns:
            results_dict['retFound 3D'][f'{oct2ir_prefix} {precision_prefix}{col}'] = row[col]
    for idx, row in retfound_3d_ir_df.iterrows():
        for col in retfound_3d_ir_df.columns:
            results_dict['retFound 3D'][f'{ir2oct_prefix} {precision_prefix}{col}'] = row[col]
    results_dict['retFound 2D'] = {}
    for idx, row in retfound_2d_df.iterrows():
        for col in retfound_2d_df.columns:
            results_dict['retFound 2D'][f'{oct2ir_prefix} {precision_prefix}{col}'] = row[col]
    for idx, row in retfound_2d_ir_df.iterrows():
        for col in retfound_2d_ir_df.columns:
            results_dict['retFound 2D'][f'{ir2oct_prefix} {precision_prefix}{col}'] = row[col]
    results_df = pd.DataFrame(results_dict)
    # return the transposed dataframe
    return results_df.T



home_directory = os.path.expanduser('~') + '/'
save_file_dir = os.path.dirname(os.path.abspath(__file__))

if 'wxdeng' in home_directory:
    home_directory = '/home/wxdeng/oph/'

retclip_aireadi_laterality_dir = home_directory + 'retclip_eval_aireadi_laterality/'
octcube_df = pd.read_csv(retclip_aireadi_laterality_dir + 'MAE3D_nodrop_laterality.csv')
octcube_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'MAE3D_nodrop_laterality_ir.csv')
retfound_3d_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound3D_laterality.csv')
retfound_2d_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound2D_laterality.csv')
retfound_3d_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound3D_laterality_ir.csv')
retfound_2d_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound2D_laterality_ir.csv')
print(octcube_df)
retclip_exp_res_laterality_aireadi_df = get_results_df(octcube_df, retfound_3d_df, retfound_2d_df, octcube_ir_df, retfound_3d_ir_df, retfound_2d_ir_df)



retclip_aireadi_laterality_dir = home_directory + 'retclip_eval_multitask_laterality/'
octcube_df = pd.read_csv(retclip_aireadi_laterality_dir + 'MAE3D_nodrop_all_laterality.csv')
octcube_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'MAE3D_nodrop_all_laterality_ir.csv')
retfound_3d_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound3D_all_laterality.csv')
retfound_2d_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound2D_all_laterality.csv')
retfound_3d_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound3D_all_laterality_ir.csv')
retfound_2d_ir_df = pd.read_csv(retclip_aireadi_laterality_dir + 'retFound2D_all_laterality_ir.csv')

retclip_exp_res_laterality_df = get_results_df(octcube_df, retfound_3d_df, retfound_2d_df, octcube_ir_df, retfound_3d_ir_df, retfound_2d_ir_df)

print(retclip_exp_res_laterality_df)
print(retclip_exp_res_laterality_aireadi_df)