
import os
from heapq import merge
import pandas as pd
import seaborn as sns

from fig_settings import *
from fig_utils import *
import cv2
from scipy.stats import ttest_rel, ttest_ind
import pickle as pkl
import matplotlib.patches as patches


home_directory = os.path.expanduser('~') + '/'
# test_cam_dir = home_directory + '/test_cam/'
# test_cam_2_dir = home_directory + '/test_cam_2/'
# test_cam_3d_dir = home_directory + '/test_cam_3d/'
# test_cam_3d_2_dir = home_directory + '/test_cam_3d_2/'
retFig_dir = home_directory + '/retFig/'
save_dir = home_directory + '/retFig/save_figs/'
patient_id = '01185f3d743cb986ea4b4774ee5d094dd8f8c618524f725ed57537d00883f53f'
visit_hash = '9b97feb68b83b543a9c12eb628982b537c32f1e8cb08da6ce4b559afaa8d6d6a'
patient_idx = 4
# patient_id = '027225ebb8dfb89f062d420f8932a8111b1adc5c0b7875ed0b34c758c0094739'
# visit_hash = 'd021710febe2006eb9b1529e07a4da04b75f425387f99abf6328e02c761bf012'

# patient_id = '0463b42fd6c055ea307ccef9ced0d701815167d255ede2e2ecaee3bd7f2867fa'
# visit_hash = '9226d810ef97858f869f0acef68dae10d462aa75309dac699e0a07303d115925'
# patient_idx = 16
# patient_id = '04c39cf8010bdff861959facf315273bbbb68b70df1f271e1272e08bbe510c0b'
# visit_hash = 'd394a756c6a8f55f315deb71d57ad3ae2f10959eabebbfc16c55a4d94bff6468'
# patient_idx = 19
patient_id = '05fb8e5c2e264a46b4751643b4319964151c4224ea7b3b3e51ea3437de1bb888'
visit_hash = '059ff1c0b92cf55f503d782ac9ff5f567f5e44fb1d8e63a550cfe543566f1f43'
patient_idx = 22

# ['01185f3d743cb986ea4b4774ee5d094dd8f8c618524f725ed57537d00883f53f', '642bcc1feb71d2c16fff6e2dd94ed73dac684f4f3648e1ac5fa842ba00c2804e']
# ['027225ebb8dfb89f062d420f8932a8111b1adc5c0b7875ed0b34c758c0094739', 'd021710febe2006eb9b1529e07a4da04b75f425387f99abf6328e02c761bf012']
# '0463b42fd6c055ea307ccef9ced0d701815167d255ede2e2ecaee3bd7f2867fa', '9226d810ef97858f869f0acef68dae10d462aa75309dac699e0a07303d115925'
# ['04c39cf8010bdff861959facf315273bbbb68b70df1f271e1272e08bbe510c0b', 'd394a756c6a8f55f315deb71d57ad3ae2f10959eabebbfc16c55a4d94bff6468']
# ['05fb8e5c2e264a46b4751643b4319964151c4224ea7b3b3e51ea3437de1bb888', '059ff1c0b92cf55f503d782ac9ff5f567f5e44fb1d8e63a550cfe543566f1f43']
Ophthal_dir = home_directory + '/Ophthal/'
case_dir = Ophthal_dir + f'/{patient_id}/macOCT/{visit_hash}/'
ir_img_path = case_dir + 'ir.png'
oct_json_path = case_dir + 'oct.json'

pred_dir = home_directory + '/retfound_baseline/outputs_ft_0710_ckpt_flash_attn/finetune_inhouse_BCLS_AMD_2DCenter_correct_patient_retfound/frame_inference_results.pkl'

with open(pred_dir, 'rb') as f:
    data = pkl.load(f)
    print(len(data))
    print(len(data[0]), data[0][0])
    print(len(data[1]), data[1][0])
    print(len(data[2]), data[2][0].shape)
    predict_logits = np.concatenate(data[2])
    predict_logits = np.reshape(predict_logits, (len(data[0]), 61, 2))
    print(predict_logits.shape)
    print(predict_logits[0])
    print(len(data[4]), data[4][0])
    labels = np.reshape(data[3], (len(data[0]), 61))
    print(labels.shape)
    
num_samples = len(data[0])
pos_prob_list = []
neg_prob_list = []
pos_patient_list = []
prob_list = []
for i in range(num_samples):
    predict_logit = predict_logits[i]
    label = labels[i][0]
    if label == 1:
        print(predict_logit)
        pos_prob_list.append(predict_logit[:,1])
        pos_patient_list.append([data[0][i], data[1][i]])
        # break
    else:
        neg_prob_list.append(predict_logit[:, 1])
    prob_list.append(predict_logit[:, 1])
print(len(pos_prob_list), len(neg_prob_list))

print(pos_patient_list[11])
print(pos_prob_list[11])
# exit()


# image_index = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
image_index = [26, 28, 30, 32, 34]
if patient_idx == 22:
    # image_index = [18, 20, 22, 24, 26, 28, 29, 30, 31, 32, 34, 36, 38, 40, 42]
    image_index = [28, 29, 30, 31, 32]
    # image_index = [i-1 for i in image_index]

original_img = []

for i in image_index:
    # original_img_path = test_cam_dir + f'original/test_cam_{i}.jpg'
    # gradcam_img_path = test_cam_dir + f'/test_cam_{i}.jpg'
    original_img_path = case_dir + f'oct-{i:03d}.png'
    original_img.append(cv2.imread(original_img_path))
    # gradcam_img.append(cv2.imread(gradcam_img_path))

    
    # gradcam_3d_img_path = test_cam_3d_dir + f'/test_cam_{i}.jpg'
    # gradcam_3d_img.append(cv2.imread(gradcam_3d_img_path))

num_frames = 5
# plot
fig, axs = plt.subplots(1, num_frames, figsize=(50, 7.6))
for i in range(num_frames):
    if patient_idx == 4:
        x, y, w, h = 350, 160, 80, 60
    if patient_idx == 22:
        x, y, w, h = 310, 200, 80, 60
    axs[i].imshow(original_img[i][y:y+h, x:x+w], vmin=0, vmax=255)


    axs[i].axis('off')
    axs[i].set_title(f'Slice {image_index[i]} (probability: {pos_prob_list[patient_idx][image_index[i]]:.2f})', fontsize=45)
    # axs[i].imshow(original_img[i+num_frames])
    # axs[i].axis('off')
    # axs[i].set_title(f'Slice {image_index[i+num_frames]}, disease probability: {pos_prob_list[patient_idx][image_index[i+num_frames]]:.3f}', fontsize=15)
    # axs[i].imshow(original_img[i+2*num_frames])
    # axs[i].axis('off')
    # axs[i].set_title(f'Slice {image_index[i+2*num_frames]}, disease probability: {pos_prob_list[patient_idx][image_index[i+2*num_frames]]:.3f}', fontsize=15)

add_patch = True
if add_patch:
    if patient_idx == 4:
        # Create a Rectangle patch
        # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
        rect = patches.Rectangle((350, 160), 80, 60, linewidth=3, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axs[ 1].add_patch(rect)
    elif patient_idx == 22:
        # Create a Rectangle patch
        rect = patches.Rectangle((310, 200), 80, 60, linewidth=3, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axs[ 3].add_patch(rect)
if patient_idx == 4:
    idx = 1
elif patient_idx == 22:
    idx = 2 
# fig.suptitle(f'AMD patient {idx}', fontsize=50)
# Adjust subplot parameters manually (you might need to tweak these values)
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=3, wspace=0.01)

# Apply tight layout with custom padding
fig.tight_layout()
plt.savefig(save_dir + f'show_slice_{patient_idx}_onecol_redbox.png')
plt.savefig(save_dir + f'show_slice_{patient_idx}_onecol_redbox.pdf', dpi=300)
plt.show()