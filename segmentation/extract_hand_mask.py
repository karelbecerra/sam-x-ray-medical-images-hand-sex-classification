import os
import sys
import torch
import numpy as np
import segment_anything
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
from matplotlib import pyplot as plt
from segment_utils import preprocess_image
from segment_utils import get_hand_mask, get_mask_image, save_all_masks
from datetime import datetime, date
import cv2

import matplotlib
matplotlib.use('TkAgg')

def check_extreme_cases(distr, images):
    idx_min = np.argmin(np.array(distr))
    idx_max = np.argmax(np.array(distr))
    return (distr[idx_min], images[idx_min]), (distr[idx_max], images[idx_max])

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = sys.argv[1] if len(sys.argv) > 1 else None
assert dataset=='tra' or dataset=='val' or dataset=='tst' or dataset=='tst_hard', 'Command line argument, wrong'

# SAM model preparation
MODEL_WEIGHTS = 'sam_vit_h_4b8939.pth'
CHKPT_PATH = '/sam/weights/'
WEIGHTS_PATH = os.path.join(CHKPT_PATH, MODEL_WEIGHTS)
MODEL_TYPE = 'vit_h'
sam = sam_model_registry[MODEL_TYPE](checkpoint=WEIGHTS_PATH)
sam.to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

# input and output data paths
base_dir = '/rsna-boneage/images/'
original_dir = os.path.join(base_dir, 'original', dataset)
mask_base_dir = os.path.join(base_dir, 'masks')
mask_dir = os.path.join(mask_base_dir, dataset)

img_size = (300, 400)

image_files = [f for f in os.listdir(original_dir)]

# order of enhancement functions to search for the hand mask
preprocess_list = ['original', 'he', 'sharp', 'gb']

# statistics to understand how the strategy work
stats = {
    'prep_count': [0]*len(preprocess_list),
    'FN': [],
    'area_distr': [],
    'solidity_distr': [],
    'conv_defect_distr': [],
    'winner_rule': {
        'areas': 0,
        'inner centroids': 0,
        'solidity': 0,
        'convexity defects': 0,
    },
    'img_id': []
}

hand_min_area, hand_max_area = (0, 0), (0, 0)
hand_min_solid, hand_max_solid = (0, 0), (0, 0)
hand_min_convdef, hand_max_convdef = (0, 0), (0, 0)

params = {
    'min_area': 0.05,
    'max_area': 0.55,  # 0.55
    'max_solidity': 0.95,  # 0.95
    'min_conv_defect': 3e5  # 3e5
}

# iterates over all images
for i, f in enumerate(image_files):
    print('\nprocessing...{} ({})'.format(f, i))
    image_path = os.path.join(original_dir, f)
    image_bgr = cv2.imread(image_path)
    prep_img = preprocess_image(image_bgr, img_size, preprocess_list)
    img_id = int(f.split('.')[0])

    j = 0
    mask = None

    # iterate over all enhanced versions of the image
    # when a mask is found, the search is stopped
    while j<len(prep_img) and mask is None:
        all_masks = mask_generator.generate(prep_img[j])
        mask, area, solidity_score, conv_defect_score, rule_id = get_hand_mask(
            all_masks,
            lower_area_perc = params['min_area'],
            upper_area_perc = params['max_area'],
            solidity_threshold = params['max_solidity'],
            conv_defect_threshold = params['min_conv_defect']
        )
        j += 1

    # if a mask was found (positive detection), statistics are recorded
    if mask is not None:
        print('Silhouette found in:', preprocess_list[-j])
        stats['prep_count'][j-1] += 1
        stats['area_distr'].append(area)
        stats['solidity_distr'].append(solidity_score)
        stats['conv_defect_distr'].append(conv_defect_score)
        stats['winner_rule'][rule_id] += 1
        stats['img_id'].append(img_id)
        # save_all_masks(all_masks, (img_size[1], img_size[0]), mask_dir, img_id)
    else:
        stats['FN'].append(img_id)

    mask_image = get_mask_image(mask, (img_size[1], img_size[0]))

    mask_path = os.path.join(mask_dir, f)
    cv2.imwrite(mask_path, mask_image * 255)

if len(stats['area_distr'])>0:
    hand_min_area, hand_max_area = check_extreme_cases(stats['area_distr'], stats['img_id'])
    hand_min_solid, hand_max_solid = check_extreme_cases(stats['solidity_distr'], stats['img_id'])
    hand_min_convdef, hand_max_convdef = check_extreme_cases(stats['conv_defect_distr'], stats['img_id'])

summary_file = os.path.join(mask_base_dir, 'results_segmentation_{}'.format(dataset), 'summary_{}.txt'.format(dataset))

today = date.today().strftime("%d/%m/%Y")
now = datetime.now().strftime("%H:%M:%S")
with open(summary_file, 'a') as f:
    f.write('\n'+'{} - {}'.format(today, now)+'\n')
    f.write("Parameterization\n")
    f.write('; '.join(preprocess_list)+'\n')
    f.write('; '.join(['{}:{}'.format(k, v) for k, v in params.items()])+'\n')
    f.write('No. of FN: {}'.format(len(stats['FN']))+'\n')
    f.write('FN:'+';'.join([str(id) for id in stats['FN']])+'\n')
    preprocess_list.reverse()
    f.write('; '.join(['{}:{}'.format(preprocess_list[i], stats['prep_count'][i]) for i in range(len(preprocess_list))])+'\n')
    f.write('; '.join(['{}:{}'.format(k, v) for k, v in stats['winner_rule'].items()])+'\n')
    f.write('Min area: {} from {}\nMax area: {} from {}\n'.format(hand_min_area[0], hand_min_area[1], hand_max_area[0], hand_max_area[1]))
    f.write('Min solidity: {} from {}\nMax solidity: {} from {}\n'.format(hand_min_solid[0], hand_min_solid[1], hand_max_solid[0], hand_max_solid[1]))
    f.write('Min convex defects: {} from {}\nMax convex defects: {} from {}\n'.format(hand_min_convdef[0], hand_min_convdef[1], hand_max_convdef[0], hand_max_convdef[1]))

distr_file = os.path.join(mask_base_dir, 'results_segmentation_{}'.format(dataset), 'distr_{}.npz'.format(dataset))
np.savez(distr_file,
    area = np.array(stats['area_distr']),
    solidity = np.array(stats['solidity_distr']),
    conv_defect = np.array(stats['conv_defect_distr']),
    img_id = np.array(stats['img_id'])
)

plt.rcParams.update({'font.size': 14})
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), layout='constrained')
titles = ['area', 'solidity', 'convexity defect']
distr = [stats['area_distr'], stats['solidity_distr'], stats['conv_defect_distr']]
x_range = [[0, 1], [0, 1], [0, 3e6]]

for i in range(len(titles)):
    axes[i].hist(distr[i], bins=20, range=(x_range[i][0], x_range[i][1]), density='true', edgecolor='gray', alpha=0.75)
    axes[i].set_xlim(x_range[i][0], x_range[i][1])
    axes[i].set_xticks(np.arange(x_range[i][0], x_range[i][1]+0.1, (x_range[i][1]-x_range[i][0])/10))
    axes[i].set_title(titles[i])

hist_img_path = os.path.join(mask_base_dir, 'results_segmentation_{}'.format(dataset), 'hist_{}.png'.format(dataset))
plt.savefig(hist_img_path, dpi=300, bbox_inches='tight')
plt.show()
