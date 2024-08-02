import os
import cv2
import pickle as pkl 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import matplotlib as mpl
import matplotlib.lines as mlines
import json
import matplotlib.patches as patches
# import torch


home_directory = os.path.expanduser('~') + '/'

retFig_dir = home_directory + '/retFig/'
save_dir = home_directory + '/retFig/save_figs/'
patient_id = '0071d4226cef124b8925360f689172f5c17a37ee9215d3b0b3e6d58aad79fdad'
visit_hash = '5f5408045df9ebf975b1fb0f312e347b7688ba5f40f7cf31b4a6250178fba024'
patient_id = '05fb8e5c2e264a46b4751643b4319964151c4224ea7b3b3e51ea3437de1bb888'
visit_hash = '059ff1c0b92cf55f503d782ac9ff5f567f5e44fb1d8e63a550cfe543566f1f43'
patient_id = '01185f3d743cb986ea4b4774ee5d094dd8f8c618524f725ed57537d00883f53f'
visit_hash = '9b97feb68b83b543a9c12eb628982b537c32f1e8cb08da6ce4b559afaa8d6d6a'
test_cam_id = 0
test_cam_dir = home_directory + f'/test_cam_{test_cam_id}/'
test_cam_2_dir = home_directory + f'/test_cam_2_{test_cam_id}/'
test_cam_3d_dir = home_directory + f'/test_cam_3d_{test_cam_id}/'
test_cam_3d_2_dir = home_directory + f'/test_cam_3d_2_{test_cam_id}/'
Ophthal_dir = home_directory + '/Ophthal/'
case_dir = Ophthal_dir + f'/{patient_id}/macOCT/{visit_hash}/'
ir_img_path = case_dir + 'ir.png'
oct_json_path = case_dir + 'oct.json'
# Load the OCT json file
with open(oct_json_path, 'r') as f:
    oct_json = json.load(f)
    print(oct_json.keys())
    octh = oct_json['octh']
    print(octh)
    print(len(octh))
    # exit()
# Load the IR image
ir_img = cv2.imread(ir_img_path)
def interpolate_line(octh, index, max_idx):
    # Extract endpoints from the first and last lines
    x1_start, y1_start, x2_start, y2_start = octh[0]
    x1_end, y1_end, x2_end, y2_end = octh[-1]

    new_line_start = [x1_start, x1_end, y1_start, y1_end]
    new_line_end = [x2_start, x2_end, y2_start, y2_end]
    # Linear interpolation between the coordinates of the start and end lines
    def interpolate(start, end, t):
        return start + t * (end - start)

    t = index / max_idx  # Normalized position of the index

    # Interpolated coordinates
    x1 = interpolate(x1_start, x2_start, t)
    x2 = interpolate(x1_end, x2_end, t)
    y1 = interpolate(y1_start, y2_start, t)
    y2 = interpolate(y1_end, y2_end, t)


    return (x1, y1, x2, y2)

# Example usage:
# octh = np.random.rand(61, 4) * 768  # Example data
index = 20  # Specific index for which the interpolated line coordinates are required
max_idx = 50  # The maximum index
interpolated_line = interpolate_line(octh, index, max_idx)
print(interpolated_line)


# image_index = [18, 22, 26, 30, 34, 38, 42]
image_index = [26, 28, 30, 32, 34]
num_frames = 5

original_img = []
gradcam_img = []
gradcam_3d_img = []
for i in image_index:
    original_img_path = test_cam_dir + f'original/test_cam_{i}.jpg'
    gradcam_img_path = test_cam_dir + f'/test_cam_{i}.jpg'
    original_img.append(cv2.imread(original_img_path))
    gradcam_img.append(cv2.imread(gradcam_img_path))

    
    gradcam_3d_img_path = test_cam_3d_dir + f'/test_cam_{i}.jpg'
    gradcam_3d_img.append(cv2.imread(gradcam_3d_img_path))


# plot
fig, axs = plt.subplots(2, num_frames, figsize=(13, 5))
# axs[0, 0].set_ylabel('IR en face image', fontsize=18)
axs[0, 0].set_ylabel('OCT slice', fontsize=18)
# axs[2, 0].set_ylabel('Saliency map\n (RETFound)', fontsize=18)
axs[1, 0].set_ylabel('Saliency map\n (OCTCube)', fontsize=18)
for i in range(num_frames):
    axs[0, i].set_title(f'Slice {image_index[i]}', fontsize=18)#, dim (y, z)')

    axs[0, i].imshow(cv2.cvtColor(original_img[i], cv2.COLOR_BGR2RGB))
    axs[0, i].set_xticks([])  # Turn off x-axis ticks
    axs[0, i].set_yticks([])  # Turn off y-axis ticks
    # axs[0, i].axis('off')
    
    # axs[2, i].imshow(cv2.cvtColor(gradcam_img[i], cv2.COLOR_BGR2RGB))
    # axs[2, i].set_xticks([])  # Turn off x-axis ticks
    # axs[2, i].set_yticks([])  # Turn off y-axis ticks
    # # axs[1, i].axis('off')
    axs[1, i].imshow(cv2.cvtColor(gradcam_3d_img[i], cv2.COLOR_BGR2RGB))
    axs[1, i].set_xticks([])  # Turn off x-axis ticks
    axs[1, i].set_yticks([])  # Turn off y-axis ticks
    # axs[2, i].axis('off')
    
    # axs[0, i].imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
    # axs[0, i].set_xticks([])  # Turn off x-axis ticks
    # axs[0, i].set_yticks([])  # Turn off y-axis ticks
    # line = octh[image_index[i]]
    # x1, y1, x2, y2 = line
    # k = (y2 - y1) / (x2 - x1)
    # if x1 < 0:
    #     x1 = 0
    #     y1 = y2 - k * x2
    # if y1 < 0:
    #     y1 = 0
    #     x1 = x2 - y2 / k
    # if x2 < 0:
    #     x2 = 0
    #     y2 = y1 + k * x1
    # if y2 < 0:
    #     y2 = 0
    #     x2 = x1 + y1 / k
    # if x1 > 768:
    #     x1 = 768
    #     y1 = y2 - k * x2
    # if y1 > 768:
    #     y1 = 768
    #     x1 = x2 - (y2 - 768) / k
    # if x2 > 767:
    #     x2 = 767
    #     y2 = y1 + k * x1
    # if y2 > 767:
    #     y2 = 767
    #     x2 = x1 + (767 - y1) / k
    # axs[0, i].plot([y1, y2], [x1, x2], 'r-', linewidth=7, alpha=0.5)
    if i == 1:
        h = 80
        w = 80
        rect = patches.Rectangle((w, h), 70, 40, linewidth=3, edgecolor='r', facecolor='none')
        axs[0, i].add_patch(rect)
        rect = patches.Rectangle((w, h), 70, 40, linewidth=3, edgecolor='r', facecolor='none')
        axs[1, i].add_patch(rect)
        # rect = patches.Rectangle((w, h), 40, 40, linewidth=3, edgecolor='r', facecolor='none')
        # axs[3, i].add_patch(rect)
    # line_start = octh[0]
    # x1, y1, x2, y2 = line_start
    # axs[3,i].plot([y1, y2], [x1, x2], 'r-', linewidth=2)
    # line_end = octh[-1]
    # x1, y1, x2, y2 = line_end
    # axs[3, i].plot([y1, y2], [x1, x2], 'b-', linewidth=2)

# cam_2_image_index = [108, 116]
# for i in range(2):
#     test_cam_2_original_img_path = test_cam_2_dir + f'original/test_cam_{cam_2_image_index[i]}.jpg'
#     test_cam_2_gradcam_img_path = test_cam_2_dir + f'/test_cam_{cam_2_image_index[i]}.jpg'
#     test_cam_3d_2_original_img_path = test_cam_3d_2_dir + f'original/test_cam_{cam_2_image_index[i]}.jpg'
#     test_cam_3d_2_gradcam_img_path = test_cam_3d_2_dir + f'/test_cam_{cam_2_image_index[i]}.jpg'
#     test_cam_2_original_img = cv2.imread(test_cam_2_original_img_path)
#     test_cam_2_gradcam_img = cv2.imread(test_cam_2_gradcam_img_path)
#     axs[0, num_frames - 2 + i].set_title(f'Frame {cam_2_image_index[i]}')#, dim (x, z)')


#     axs[1, num_frames - 2 + i].imshow(cv2.cvtColor(test_cam_2_original_img, cv2.COLOR_BGR2RGB))
#     axs[1, num_frames - 2 + i].set_xticks([])  # Turn off x-axis ticks
#     axs[1, num_frames - 2 + i].set_yticks([])  # Turn off y-axis ticks
#     # axs[0, num_frames - 2 + i].axis('off')

#     axs[2, num_frames - 2 + i].imshow(cv2.cvtColor(test_cam_2_gradcam_img, cv2.COLOR_BGR2RGB))
#     axs[2, num_frames - 2 + i].set_xticks([])  # Turn off x-axis ticks
#     axs[2, num_frames - 2 + i].set_yticks([])  # Turn off y-axis ticks
#     # axs[1, num_frames - 2 + i].axis('off')

#     axs[3, num_frames - 2 + i].imshow(cv2.cvtColor(cv2.imread(test_cam_3d_2_gradcam_img_path), cv2.COLOR_BGR2RGB))
#     axs[3, num_frames - 2 + i].set_xticks([])  # Turn off x-axis ticks
#     axs[3, num_frames - 2 + i].set_yticks([])  # Turn off y-axis ticks
#     # axs[2, num_frames - 2 + i].axis('off')

#     axs[0, num_frames - 2 + i].imshow(cv2.cvtColor(ir_img, cv2.COLOR_BGR2RGB))
#     axs[0, num_frames - 2 + i].set_xticks([])  # Turn off x-axis ticks
#     axs[0, num_frames - 2 + i].set_yticks([])  # Turn off y-axis ticks
#     line = interpolate_line(octh, cam_2_image_index[i], 256)
#     x1, y1, x2, y2 = line
#     k = (y2 - y1) / (x2 - x1)
#     if x1 < 0:
#         x1 = 0
#         y1 = y2 - k * x2
#     if y1 < 0:
#         y1 = 0
#         x1 = x2 - y2 / k
#     if x2 < 0:
#         x2 = 0
#         y2 = y1 + k * x1
#     if y2 < 0:
#         y2 = 0
#         x2 = x1 + y1 / k
#     if x1 > 767:
#         x1 = 767
#         y1 = y2 - k * x2
#     if y1 > 767:
#         y1 = 767
#         x1 = x2 - (y2 - 768) / k
#     if x2 > 767:
#         x2 = 767
#         y2 = y1 + k * x1
#     if y2 > 767:
#         y2 = 767
#         x2 = x1 + (767 - y1) / k

#     axs[0, num_frames - 2 + i].plot([y1, y2], [x1, x2], 'r-', linewidth=1)

# Use fig.text to add global annotations
# fig.text(0.053, 0.68, 'First 7 columns: \n Dimension (y, z)', ha='center', va='center', fontsize=12, color='red')
# fig.text(0.782, 0.68, 'Last 2 columns: \n Dimension (x, z)', ha='center', va='center', fontsize=12, color='red')

# # use rainbow color map
# cmap = mpl.cm.rainbow

# # Define the normalization from 0 to 1
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# # line_position = 7.5 / num_frames  # Position between the 7th and 8th frame, adjust as necessary
# # plt.axvline(x=line_position, color='grey', linestyle='--', linewidth=2, )

# # Adjust the layout manually to leave space for the colorbar
# # plt.subplots_adjust(right=0.88, top=0.85, bottom=0.15, left=0.05)
# # Apply tight layout first with space for the colorbar
fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0, rect=[0, 0, 0.995, 1])  # Leave space on the right


# # Add an additional axis at the right of the figure for the colorbar
# cbar_ax = fig.add_axes([0.97, 0.01, 0.02, 0.47])  # Adjust the left value to prevent overlapping

# # Create and display the colorbar
# cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation='vertical',)
# # Misuse the ylabel to act as the colorbar label
# # cbar.ax.set_xlabel('Saliency', labelpad=10, rotation=0, fontsize=15)
# # cbar.set_label('Saliency', labelpad=0, rotation=0, verticalalignment='top')
# # cbar.set_label('Saliency', labelpad=10, rotation=270, va='top')
# # # Adjust layout to make sure nothing is clipped and labels are visible
# # cbar.set_label('Saliency', labelpad=10)
# # cbar.ax.xaxis.set_label_position('top')
# # cbar.ax.xaxis.set_ticks_position('top')
# # plt.colorbar(vmin=0, vmax=1, cmap='rainbow')
# cbar.ax.yaxis.set_ticks([])

line_position = (7 / num_frames)  # Adjust according to your specific layout
# Adjusting x position according to the figure size
x_position = line_position * (fig.get_size_inches()[0] * fig.dpi)
# fig.text(0.962, 0.5, r'($\uparrow$)', fontsize=20)


# Adding the line to the figure
line = mlines.Line2D([x_position, x_position], [0, 1], transform=fig.transFigure, color="gray", linestyle="--", linewidth=2)
fig.add_artist(line)
# fig.tight_layout()
plt.savefig(save_dir + f'gradcam_vis_plot_xz_nocbar_supp_fig1_{test_cam_id}_tworow.jpg')
plt.savefig(save_dir + f'gradcam_vis_plot_xz_nocbar_supp_fig1_{test_cam_id}_tworow.pdf', format='pdf', dpi=300)



# '0071d4226cef124b8925360f689172f5c17a37ee9215d3b0b3e6d58aad79fdad', '5f5408045df9ebf975b1fb0f312e347b7688ba5f40f7cf31b4a6250178fba024'