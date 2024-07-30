
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






patient_id = 'f509349545c3f6d1d871f39373335dd4250f2ecaa03379daa34f50fc7e503b7f'
# visit_hash = '8bd87146c0abc05fb88ea23b58f49a18d5401191139740862915d4e258d0faf0'
# # visit_hash = '2377255a873510e7352cecce5eeee79fca54f53c36dd967bbc188422249bef73'
# patient_idx = 848
# title = 'Diabetes patient OS'
visit_hash = '859c5e5091f29a474581ac321c2c37c47a0535dc53a98169afdf2588a823cb0c'
patient_idx = 8490
title = 'Diabetes patient OD 1 year'
pred_dir = home_directory + '/retfound_baseline/non_oph_outputs_ft_2d_0724_more_runs_ckpt_flash_attn/runs_1/l1_100_finetune_inhouse_multi_label_2DCenter_correct_patient_singlefold_retfound/frame_inference_results_temp.pkl'
with open(pred_dir, 'rb') as f:
    data = pkl.load(f)
    predict_logit_8490 = np.array(data[2][:61])
    patient_id = data[0][0]
    visit_hash = data[1][0]
    print(patient_id, visit_hash)
    print(predict_logit_8490.shape, predict_logit_8490[:, 9], )
    predict_logit_8490_diabete = predict_logit_8490[:, 9]
    # exit()


# patient_id = '027225ebb8dfb89f062d420f8932a8111b1adc5c0b7875ed0b34c758c0094739'
# visit_hash = 'd021710febe2006eb9b1529e07a4da04b75f425387f99abf6328e02c761bf012'

# patient_id = '0463b42fd6c055ea307ccef9ced0d701815167d255ede2e2ecaee3bd7f2867fa'
# visit_hash = '9226d810ef97858f869f0acef68dae10d462aa75309dac699e0a07303d115925'
# patient_idx = 16
# patient_id = '04c39cf8010bdff861959facf315273bbbb68b70df1f271e1272e08bbe510c0b'
# visit_hash = 'd394a756c6a8f55f315deb71d57ad3ae2f10959eabebbfc16c55a4d94bff6468'
# patient_idx = 19

# patient_id = '05fb8e5c2e264a46b4751643b4319964151c4224ea7b3b3e51ea3437de1bb888'
# visit_hash = '059ff1c0b92cf55f503d782ac9ff5f567f5e44fb1d8e63a550cfe543566f1f43'
# patient_idx = 22

# ['01185f3d743cb986ea4b4774ee5d094dd8f8c618524f725ed57537d00883f53f', '642bcc1feb71d2c16fff6e2dd94ed73dac684f4f3648e1ac5fa842ba00c2804e']
# ['027225ebb8dfb89f062d420f8932a8111b1adc5c0b7875ed0b34c758c0094739', 'd021710febe2006eb9b1529e07a4da04b75f425387f99abf6328e02c761bf012']
# '0463b42fd6c055ea307ccef9ced0d701815167d255ede2e2ecaee3bd7f2867fa', '9226d810ef97858f869f0acef68dae10d462aa75309dac699e0a07303d115925'
# ['04c39cf8010bdff861959facf315273bbbb68b70df1f271e1272e08bbe510c0b', 'd394a756c6a8f55f315deb71d57ad3ae2f10959eabebbfc16c55a4d94bff6468']
# ['05fb8e5c2e264a46b4751643b4319964151c4224ea7b3b3e51ea3437de1bb888', '059ff1c0b92cf55f503d782ac9ff5f567f5e44fb1d8e63a550cfe543566f1f43']


patient_id = 'f509349545c3f6d1d871f39373335dd4250f2ecaa03379daa34f50fc7e503b7f'
visit_hash = 'b1a401a8aac6a91e5e947f6e1ce662951efba436808df463a0c1382e4fcad8ba'
patient_idx = 849
title = 'Diabetes patient OD'
Ophthal_dir = home_directory + '/Ophthal/'
case_dir = Ophthal_dir + f'/{patient_id}/macOCT/{visit_hash}/'
ir_img_path = case_dir + 'ir.png'
oct_json_path = case_dir + 'oct.json'

pred_dir = home_directory + '/retfound_baseline/non_oph_outputs_ft_2d_0724_more_runs_ckpt_flash_attn/runs_1/l1_100_finetune_inhouse_multi_label_2DCenter_correct_patient_singlefold_retfound/frame_inference_results.pkl'
num_classes = 455
with open(pred_dir, 'rb') as f:
    data = pkl.load(f)
    print(len(data))
    print(len(data[0]), data[0][0])
    print(len(data[1]), data[1][0])
    print(len(data[2]), data[2][0].shape)
    predict_logits = np.concatenate(data[2])
    predict_logits = np.reshape(predict_logits, (len(data[0]), 61, num_classes))
    print(predict_logits.shape)
    print(predict_logits[0])
    print(len(data[4]), data[4][0])
    true_label_onehot = np.reshape(data[4], (len(data[0]), 61, num_classes))
    labels = np.reshape(data[3], (len(data[0]), 61))
    print(labels.shape)

diabete_idx = 9
num_samples = len(data[0])
pos_prob_list = []
neg_prob_list = []
pos_patient_list = []
prob_list = []
for i in range(num_samples):
    predict_logit = predict_logits[i, :, diabete_idx]
    label = true_label_onehot[i, 0, diabete_idx]
    if label == 1:
        print(predict_logit)
        pos_prob_list.append(predict_logit[:])
        pos_patient_list.append([data[0][i], data[1][i]])
        # break
    else:
        neg_prob_list.append(predict_logit[:])
    prob_list.append(predict_logit[:])
print(len(pos_prob_list), len(neg_prob_list))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(predict_logit_8490_diabete, label='OD 1 year later')
ax.plot(pos_prob_list[patient_idx], label='OD current')
ax.set_xlabel('Slice index', fontsize=15)
ax.set_ylabel('Probability', fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.tick_params(axis='both', which='major', labelsize=15)
fig.legend(ncol=2, loc='upper center', frameon=False, fontsize=15)
plt.savefig(save_dir + f'prob_case_diabetes.png')
exit()



# print(pos_patient_list[patient_idx])
# print(pos_prob_list[patient_idx])
# exit()


image_index = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44]
if patient_idx == 22:
    image_index = [18, 20, 22, 24, 26, 28, 29, 30, 31, 32, 34, 36, 38, 40, 42]
    # image_index = [i-1 for i in image_index]

image_index = [26, 30,]

original_img = []

for i in image_index:
    # original_img_path = test_cam_dir + f'original/test_cam_{i}.jpg'
    # gradcam_img_path = test_cam_dir + f'/test_cam_{i}.jpg'
    original_img_path = case_dir + f'oct-{i:03d}.png'
    original_img.append(cv2.imread(original_img_path))
    # gradcam_img.append(cv2.imread(gradcam_img_path))

    
    # gradcam_3d_img_path = test_cam_3d_dir + f'/test_cam_{i}.jpg'
    # gradcam_3d_img.append(cv2.imread(gradcam_3d_img_path))

num_frames = 1
# plot
fig, axs = plt.subplots(num_frames, 2, figsize=(80/3, 9))
if patient_idx == 8490:
    for i in range(2):
        axs[i].imshow(original_img[i])
        axs[i].axis('off')
        axs[i].set_title(f'Slice {image_index[i]}, (probability: {predict_logit_8490_diabete[image_index[i]]:.2f})', fontsize=30)
        # axs[i].imshow(original_img[i+num_frames])
        # axs[i].axis('off')
        # axs[i].set_title(f'Slice {image_index[i+num_frames]}, (probability: {predict_logit_8490_diabete[image_index[i]]:.2f})', fontsize=20)
        # axs[i].imshow(original_img[i+2*num_frames])
        # axs[i].axis('off')
        # axs[i].set_title(f'Slice {image_index[i+2*num_frames]}, (probability: {predict_logit_8490_diabete[image_index[i]]:.2f})', fontsize=20)
else:
    for i in range(2):
        axs[i].imshow(original_img[i])
        axs[i].axis('off')
        axs[i].set_title(f'Slice {image_index[i]}, (probability: {pos_prob_list[patient_idx][image_index[i]]:.2f})', fontsize=30)
        # axs[i].imshow(original_img[i])
        # axs[i].axis('off')
        # axs[i].set_title(f'Slice {image_index[i+num_frames]}, (probability: {pos_prob_list[patient_idx][image_index[i]]:.2f})', fontsize=20)
        # axs[i].imshow(original_img[i])
        # axs[i].axis('off')
        # axs[i].set_title(f'Slice {image_index[i+2*num_frames]}, (probability: {pos_prob_list[patient_idx][image_index[i]]:.2f})', fontsize=20)

add_patch = True
if add_patch:
    if patient_idx == 4:
        # Create a Rectangle patch
        # rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor='r', facecolor='none')
        rect = patches.Rectangle((350, 160), 80, 60, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axs[1, 1].add_patch(rect)
    elif patient_idx == 22:
        # Create a Rectangle patch
        rect = patches.Rectangle((310, 200), 80, 60, linewidth=1, edgecolor='r', facecolor='none')

        # Add the patch to the Axes
        axs[1, 3].add_patch(rect)
    elif patient_idx == 849:
        rect = patches.Rectangle((180, 100), 80, 60, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, 2].add_patch(rect)
        rect = patches.Rectangle((170, 100), 80, 60, linewidth=1, edgecolor='r', facecolor='none')
        axs[0, 4].add_patch(rect)
        rect = patches.Rectangle((290, 120), 80, 60, linewidth=1, edgecolor='r', facecolor='none')
        axs[1, 0].add_patch(rect)
        # rect = patches.Rectangle((140, 100), 80, 60, linewidth=1, edgecolor='blue', facecolor='none')
        # axs[1, 1].add_patch(rect)
        rect = patches.Rectangle((130, 100), 80, 60, linewidth=1, edgecolor='r', facecolor='none')
        axs[2, 2].add_patch(rect)
        rect = patches.Rectangle((100, 100), 80, 60, linewidth=1, edgecolor='r', facecolor='none')
        axs[2, 3].add_patch(rect)

if patient_idx == 4:
    idx = 1
elif patient_idx == 22:
    idx = 2 
else:
    idx = 0
fig.suptitle(f'{title}', fontsize=30)
plt.tight_layout()
plt.savefig(save_dir + f'diabete_show_slice_{patient_idx}_oneimage_row.png')
plt.savefig(save_dir + f'diabete_show_slice_{patient_idx}_oneimage_row.pdf', dpi=300)
plt.show()