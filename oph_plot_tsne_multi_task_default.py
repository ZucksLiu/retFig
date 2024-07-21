import os
import pickle as pkl 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import umap
# import torch

home_directory = os.path.expanduser('~') + '/'
retfound_dir = home_directory + 'retfound_baseline/'
retFig_dir = home_directory + 'retFig/'
print(home_directory)
file_path = retfound_dir + 'outputs_ft_st_0628_ckpt_flash_attn/' + 'finetune_inhouse_multi_task_default_3D_correct_patient_singlefold_3d_256_0509_inference/' + 'frame_inference_results.pkl'

with open(file_path, 'rb') as f:
    data = pkl.load(f)
    print(len(data))
    print(len(data[0]), data[0][0])
    print(len(data[1]), data[1][0])
    print(len(data[2]), data[2][0].shape)
    predict_logits = np.concatenate(data[2])
    # predict_logits = np.reshape(predict_logits, (len(data[0]), 61, 2))
    print(predict_logits.shape)
    print(predict_logits[0])
    print(len(data[4]), data[4][0])
    print(len(data[3]), data[3][0].shape)
    labels = np.array(data[4])
    print(labels.shape, labels[:100])
    # label_multi_task = np.reshape(labels, (len(data[0]), 8, 2))
    # print(label_multi_task.shape, label_multi_task[:100])
    targets = np.array(data[6])
    print('targets:', targets.shape, targets[:100])
    embedding_list = data[5]
    print(len(embedding_list), embedding_list[0].shape)
    embedding = np.array(embedding_list)
    print(embedding.shape)


freqeuncy_of_diseases_rank = [0, 2, 6, 5, 4, 3, 1, 8, 7]
onehot_targets = []
for i in range(len(targets)):
    target = targets[i]

    if sum(target) == 1:
        idx_1 = np.where(target == 1)[0][0]
        onehot_targets.append(idx_1)
    else:
        for idx in freqeuncy_of_diseases_rank[::-1]:
            if target[idx] == 1:
                onehot_targets.append(idx)
                break
print(onehot_targets[:100])


# Count unique combinations
unique_combinations, inverse_indices, counts = np.unique(targets, axis=0, return_counts=True, return_inverse=True)
sum_targets = np.sum(targets, axis=1)
print(sum(sum_targets <=2))
subset_targets = targets[sum_targets <=2 ]
print(np.unique(subset_targets, axis=0, return_counts=True), len(np.unique(subset_targets, axis=0)))
# exit()
# print(len(t), t)
#  
print(unique_combinations, counts, len(unique_combinations))
print(sum(counts>20))
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
embedding = np.array(embedding)
print(onehot_targets!=0)
# filter embedding that onehot_targets != 0
idx_filter = np.array([i for i in range(len(onehot_targets)) if onehot_targets[i] != 0], dtype=int )
embedding = embedding[idx_filter]
print(embedding.shape)
# print(onehot_targets.shape, idx_filter)
used_targets = [onehot_targets[i] for i in idx_filter]
embeddings_2d = reducer.fit_transform(embedding)

# # exit()
# if os.path.exists(retFig_dir + 'multi_task_default_embedding.npy'):
#     print('Loading precomputed t-SNE embeddings')
#     embeddings_2d = np.load(retFig_dir + 'multi_task_default_embedding.npy')
# else:
#     # Fit t-SNE
#     tsne = TSNE(n_components=2, random_state=42)
#     embeddings_2d = tsne.fit_transform(embedding)
#     # save embeddings
#     np.save(retFig_dir + 'multi_task_default_embedding.npy', embeddings_2d)
# print(embeddings_2d.shape)
# exit()
# Plotting
plt.figure(figsize=(12, 8))
# for idx, combo in enumerate(unique_combinations):
#     combo_indices = np.where(inverse_indices == idx)
#     print(combo, len(combo_indices[0]))
#     plt.scatter(embeddings_2d[combo_indices, 0], embeddings_2d[combo_indices, 1], label=str(combo), alpha=0.5)
# for idx, label in enumerate(np.unique(onehot_targets)):
    # indices = np.where(np.array(onehot_targets) == label)
for idx, label in enumerate(np.unique(used_targets)):
    indices = np.where(used_targets == label)
    # if label in [ 4, 5, 6, 7, 8]:
        # continue
    print('label:', label, len(indices[0]))
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=str(label), alpha=1)


plt.title('t-SNE of Embeddings with Multi-Label Targets')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Target Combinations', bbox_to_anchor=(1, 1.05), loc='upper left')
plt.savefig(retFig_dir + 'save_figs/multi_task_default_tsne.png')
plt.show()