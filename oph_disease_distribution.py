import os
import json
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

home_directory = os.path.expanduser('~') + '/'
Oph_cls_task_directory = home_directory + 'Oph_cls_task/'
icd10_code_dir = Oph_cls_task_directory + 'multi_label_expr_all_0529/' 
multilabel_dict_json = icd10_code_dir + 'multilabel_cls_dict.json'
out_dir = home_directory + 'retFig/save_figs/'

with open(multilabel_dict_json, 'r') as f:
    multilabel_dict = json.load(f)
    # print(multilabel_dict)
    print(len(multilabel_dict))
    print(multilabel_dict.keys())
    disease_list = multilabel_dict['disease_list']
    patient_dict = multilabel_dict['patient_dict']
    print(len(disease_list), disease_list)
    print(len(patient_dict))

num_disease = len(disease_list)
patient_num_distribution = np.zeros(num_disease)
for patient_id, disease_list_i in patient_dict.items():
    patient_num_distribution += np.array(disease_list_i)
print(patient_num_distribution)

filtered_disease_list = {}
for i, disease_icd in enumerate(disease_list.keys()):
    # print(disease_icd, disease_list[disease_icd], patient_num_distribution[i])
    if disease_icd.startswith('Z'):
        continue
    filtered_disease_list[disease_icd] = patient_num_distribution[i]
    print('Disease', disease_icd, 'Number of patients', patient_num_distribution[i])
    if len(filtered_disease_list) > 100:
        break
print(len(filtered_disease_list))
# plot the distribution of the diseases
sorted_disease_list = sorted(filtered_disease_list.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(14, 10))
plt.gcf().set_facecolor('#f0f0f0')  # Set the background color for the entire figure

# plt.bar(range(len(sorted_disease_list)), [x[1] for x in sorted_disease_list])
# plt.xticks(range(len(sorted_disease_list)), [x[0] for x in sorted_disease_list], rotation=45)
full_palette = sns.color_palette("Blues", len(sorted_disease_list)+ 2)[::-1]
palette = full_palette[:int(len(sorted_disease_list) )]  # Using only the first 80% of the Blues palette in reverse

width=0.9
bars = plt.bar(list(range(len(sorted_disease_list))), [x[1] for x in sorted_disease_list], width, color=palette) # , edgecolor='black')
# ax.set_facecolor('#f0f0f0')
plt.gca().set_facecolor('#f0f0f0')
# Adjust the ticks and labels
plt.xticks(range(len(sorted_disease_list)), [x[0] for x in sorted_disease_list], ha='center', fontsize=15)
plt.ylabel('Number of patients', fontsize=20)
plt.xlabel('Disease name abbreviation', fontsize=20)
# set font size of yticks
plt.yticks(fontsize=15)
plt.title('Distribution of retinal Diseases', fontsize=20)
# remove the top and right spines
sns.despine()


# Set the x-axis limits to start closer to the first bar
# plt.xlim(-0.5, len(sorted_disease_list) - 0.5)

# Remove blank spaces on the left and right and bring y-axis closer
plt.gca().margins(x=0.01)
plt.tight_layout()  # Reduce padding around the plot


# # Adding the inset subplot in the upper right corner
# inset_disease_num = 7
# inset_axes = plt.axes([0.55, 0.4, 0.4, 0.5])  # [left, bottom, width, height]
# inset_palette = sns.color_palette("Blues", inset_disease_num + 1)[::-1]
# inset_bars = inset_axes.bar(range(inset_disease_num), [x[1] for x in sorted_disease_list[:inset_disease_num]], color=inset_palette, width=0.9)
# inset_axes.set_xticks(range(inset_disease_num))
# inset_axes.set_xticklabels([x[0] for x in sorted_disease_list[:inset_disease_num]], ha='right', fontsize=10)
# inset_axes.set_yticks(range(0, 4001, 1000))
# inset_axes.tick_params(axis='both', which='major', labelsize=14)
# inset_axes.set_facecolor('#f0f0f0')
# inset_axes.set_ylabel('Number of patients', fontsize=18)
# # inset_axes.set_xlabel('Disease ICD-10 code', fontsize=10)
# inset_axes.set_title(f'Top {inset_disease_num} diseases', fontsize=18, y=-0.15)
# sns.despine(ax=inset_axes)


# # Remove blank spaces on the left and right
# # plt.subplots_adjust(left=0.2, right=0.99, bottom=0.2)  # Adjust left to reduce space

plt.savefig(out_dir + 'oph_disease_distribution.png')
plt.savefig(out_dir + 'oph_disease_distribution.pdf', dpi=300)