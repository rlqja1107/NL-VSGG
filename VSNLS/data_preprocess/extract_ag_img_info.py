# Refer to https://github.com/yrcong/STTran/blob/main/dataloader/action_genome.py

import os
import sys
sys.path.append('.')
import torch
import pickle
import numpy as np
from tqdm import tqdm
from cv2 import imread
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob

frames_path = 'datasets/AG/frames/'

# Train
with open("datasets/AG/ag_train_id.pkl", 'rb') as f:
    train_frame_list = pickle.load(f)

print("Extracting Train Videos' Scale Information")
im_info_dict = {}
for video_index, frame_list in tqdm(train_frame_list.items()):
    frame_list = frame_list[:1]
    processed_ims = []
    im_scales = []
    for idx, f_name in enumerate(frame_list):
        im = imread(os.path.join(frames_path+f"{video_index}", f_name)) # channel h,w,3
        im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
    im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
    im_info_dict[video_index] = im_info[0:1]


with open("datasets/AG/ag_img_info_train.pkl", 'wb') as f:
    pickle.dump(im_info_dict, f)


# Test
with open("datasets/AG/ag_test_id.pkl", 'rb') as f:
    test_frame_list = pickle.load(f)

print("Extracting Test Videos' Scale Information")
im_info_dict = {}
for video_index, frame_list in tqdm(test_frame_list.items()):
    frame_list = frame_list[:1]
    processed_ims = []
    im_scales = []

    for idx, f_name in enumerate(frame_list):
        im = imread(os.path.join(frames_path+f"{video_index}", f_name)) # channel h,w,3
        im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
        im_scales.append(im_scale)
        processed_ims.append(im)

    blob = im_list_to_blob(processed_ims)
    im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
    im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
    im_info_dict[video_index] = im_info[0:1]

with open("datasets/AG/ag_img_info_test.pkl", 'wb') as f:
    pickle.dump(im_info_dict, f)