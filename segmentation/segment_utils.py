import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import math


def apply_preprocess_function(input_img, preprocess):
    prep_img = input_img
    if preprocess == 'sharp':
        # image sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        prep_img = cv2.filter2D(input_img, -1, kernel)
    elif preprocess == 'he':
        # histogram equalization
        prep_img = cv2.equalizeHist(input_img)
    elif preprocess == 'clahe':
        clahe = cv2.createCLAHE()
        prep_img = clahe.apply(input_img)
    elif preprocess == 'gb':
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16.0
        prep_img = cv2.filter2D(input_img, -1, kernel)
    return prep_img


def preprocess_image(input_img, target_size, preprocess_order=['original', 'he', 'sharp', 'gb']):
    preprocess_imgs = []
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    for i, preprocess in enumerate(preprocess_order):
        image = apply_preprocess_function(image, preprocess)
        output_img = cv2.resize(image, target_size, interpolation= cv2.INTER_LINEAR)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2RGB)
        preprocess_imgs = [output_img] + preprocess_imgs
        # preprocess_imgs.append(output_img)
    return preprocess_imgs


def find_contours(img, returnPoints):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])
    cont = contours[np.argmax(areas)]
    hull = cv2.convexHull(cont, returnPoints=returnPoints)
    return cont, hull


def find_candidate_masks_by_area(masks, lower_bound, upper_bound):
    candidates = []
    cand_areas = []
    cand_idxs = []
    rule_id = 'areas'
    for i, m in enumerate(masks):
        if m['area']>=lower_bound and m['area']<=upper_bound:
            candidates.append(m)
            cand_areas.append(m['area'])
            cand_idxs.append(i)
    return candidates, cand_areas, cand_idxs, rule_id


def find_candidate_masks_by_inner_centroid(masks, i_size):

    x_margin = int(0.05*i_size[0])
    y_margin = int(0.05*i_size[1])

    def check_neighborhood(m, x, y):
        from_x = max(0, x-x_margin)
        to_x = min(i_size[0], x+x_margin)
        from_y = max(0, y-y_margin)
        to_y = min(i_size[1], y+y_margin)
        region_mass = np.sum(m['segmentation'][from_x:to_x,from_y:to_y])
        expected_mass = (to_x-from_x) * (to_y-from_y)
        return True if region_mass == expected_mass else False

    candidates = []
    cand_centroids = []
    cand_idxs = []
    rule_id = 'inner centroids'
    for i, m in enumerate(masks):
        # find indices with mass
        mass_x, mass_y = np.where(m['segmentation'] == True)
        x_center = round(np.average(mass_x))
        y_center = round(np.average(mass_y))
        print('Calculated center: ({} {})'.format(x_center, y_center))
        if check_neighborhood(m, x_center, y_center):
            candidates.append(m)
            cand_centroids.append((x_center, y_center))
            cand_idxs.append(i)
    return candidates, cand_centroids, cand_idxs, rule_id


def find_candidates_masks_by_low_solidity(masks, upper_bound=0.95):
    candidates = []
    cand_solidity = []
    cand_idxs = []
    rule_id = 'solidity'
    for i, m in enumerate(masks):
        cont, hull = find_contours(m['segmentation'].astype(np.uint8), True)
        cont_area = cv2.contourArea(cont)
        hull_area = cv2.contourArea(hull)
        # solidity
        s = cont_area/(hull_area+1e-10)
        if s < upper_bound:
            candidates.append(m)
            cand_solidity.append(s)
            cand_idxs.append(i)
    return candidates, cand_solidity, cand_idxs, rule_id


def find_candidate_masks_by_convexity_defects(masks, lower_bound_score=3e5, end_rule=False):
    candidates = []
    cand_defect_scores = []
    cand_idxs = []
    rule_id = 'convexity defects'
    for i, m in enumerate(masks):
        cont, hull = find_contours(m['segmentation'].astype(np.uint8), False)
        hull_flat = hull.flatten()
        if np.argmin(hull_flat) != len(hull_flat)-1:  # bug in hull extraction in 9278.png
            continue
        convexity_defects = cv2.convexityDefects(cont, hull)
        n_defects = convexity_defects.shape[0]
        def_weight = 0
        def_distr = []
        for j in range(n_defects):
            s, e, f, d = convexity_defects[j, 0]  # d is the dist to the farthest nonconvex point
            def_distr.append(d)

        def_score = np.sum(np.array(def_distr))
        print('Convexity defect stats (num def, def accum):', n_defects, def_score)
        def_score *= n_defects
        if def_score >= lower_bound_score:
            candidates.append(m)
            cand_defect_scores.append(def_score)
            cand_idxs.append(i)

    if end_rule and len(candidates)>0:
        id_max_conv_defect = np.argmax(np.array(cand_defect_scores))
        candidates = [candidates[id_max_conv_defect]]
        cand_defect_scores = [cand_defect_scores[id_max_conv_defect]]
        cand_idxs = [cand_idxs[id_max_conv_defect]]

    return candidates, cand_defect_scores, cand_idxs, rule_id  # max(defect_weights) if len(candidates)>0 else 0


def hand_mask_feat_init():
    feat = {
        'hand_mask': None,
        'hand_mask_area': 0,
        'hand_mask_solidity': 0,
        'hand_mask_conv_defect_score': 0,
        'hand_mask_chosen_by': None
    }
    return feat


def hand_mask_stat_init():
    feat = {
        'masks': None,
        'areas': None,
        'solidity': None,
        'convexity defects': None,
        'inner centroids': None
    }
    return feat


def update_stat_dict(d, idxs):
    for k in d:
        if d[k] is not None:
            d[k] = d[k][idxs]
    return d


def decode_feat_dict(fd):
    return fd['hand_mask'], fd['hand_mask_area'], fd['hand_mask_solidity'], fd['hand_mask_conv_defect_score'], fd['hand_mask_chosen_by']


def get_hand_mask(masks, lower_area_perc=0.05, upper_area_perc=0.55, solidity_threshold=0.95, conv_defect_threshold=3e5):
    # req 1: mask area >= lower_perc * image area (height * width)
    # req 2: mask centroid is part of the mask (is inside the mask)
    # req 3: mask with low solidity
    # req 4: mask with high convexity defects

    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    img_size = sorted_masks[0]['segmentation'].shape[:2]
    img_area = img_size[0] * img_size[1]
    # print('Mask areas...')
    # for i in range(min(5, len(masks))):
    #     print(sorted_masks[i]['area'])

    feat_dict = hand_mask_feat_init()
    stat_dict = hand_mask_stat_init()

    # req 1
    candidate_masks, candidate_results, candidate_idxs, rule_id = find_candidate_masks_by_area(sorted_masks, lower_area_perc*img_area, upper_area_perc*img_area)
    print('No. of candidates by area:', len(candidate_masks))

    if len(candidate_masks) == 0:
        feat_dict = hand_mask_feat_init()
        return decode_feat_dict(feat_dict)

    if feat_dict['hand_mask_chosen_by'] is None and len(candidate_masks) == 1:
        feat_dict['hand_mask_chosen_by'] = rule_id

    stat_dict = update_stat_dict(stat_dict, np.array(candidate_idxs))
    stat_dict[rule_id] = np.array(candidate_results)

    # req 2
    candidate_masks, candidate_results, candidate_idxs, rule_id = find_candidate_masks_by_inner_centroid(candidate_masks, img_size)
    print('No. of candidates by inner centroid:', len(candidate_masks))

    if len(candidate_masks) == 0:
        feat_dict = hand_mask_feat_init()
        return decode_feat_dict(feat_dict)

    if feat_dict['hand_mask_chosen_by'] is None and len(candidate_masks) == 1:
        feat_dict['hand_mask_chosen_by'] = rule_id

    stat_dict = update_stat_dict(stat_dict, np.array(candidate_idxs))
    stat_dict[rule_id] = np.array(candidate_results)

    # req 3
    candidate_masks, candidate_results, candidate_idxs, rule_id = find_candidates_masks_by_low_solidity(candidate_masks, upper_bound=solidity_threshold)
    print('No. of candidates by low solidity:', len(candidate_masks))

    if len(candidate_masks) == 0:
        feat_dict = hand_mask_feat_init()
        return decode_feat_dict(feat_dict)

    if feat_dict['hand_mask_chosen_by'] is None and len(candidate_masks) == 1:
        feat_dict['hand_mask_chosen_by'] = rule_id

    stat_dict = update_stat_dict(stat_dict, np.array(candidate_idxs))
    stat_dict[rule_id] = np.array(candidate_results)

    # req 4
    candidate_masks, candidate_results, candidate_idxs, rule_id = find_candidate_masks_by_convexity_defects(candidate_masks, lower_bound_score=conv_defect_threshold, end_rule=True)
    print('No. of candidates by high covexity defects:', len(candidate_masks))

    if len(candidate_masks) == 0:
        feat_dict = hand_mask_feat_init()
        return decode_feat_dict(feat_dict)

    if feat_dict['hand_mask_chosen_by'] is None and len(candidate_masks) == 1:
        feat_dict['hand_mask_chosen_by'] = rule_id

    stat_dict = update_stat_dict(stat_dict, np.array(candidate_idxs))
    stat_dict[rule_id] = np.array(candidate_results)

    feat_dict['hand_mask'] = candidate_masks[0]
    feat_dict['hand_mask_area'] = stat_dict['areas'][0]/img_area
    feat_dict['hand_mask_solidity'] = stat_dict['solidity'][0]
    feat_dict['hand_mask_conv_defect_score'] = stat_dict['convexity defects'][0]

    return decode_feat_dict(feat_dict)


def get_mask_image(mask, img_size):
    segment_img = np.zeros(img_size + (1,))
    if mask is not None:
        segment_img[mask['segmentation']] = 1
    return segment_img


def save_all_masks(masks, img_size, dir, id):
    for j, m in enumerate(masks):
        mask_im = np.zeros(img_size + (1,))
        mask_im[m['segmentation']] = 255
        cv2.imwrite(os.path.join(dir, '{}_{}.png'.format(id, j)), mask_im)


def show_masks(annotations, target_size):
    sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
    img = np.zeros((target_size[1], target_size[0], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    return img
