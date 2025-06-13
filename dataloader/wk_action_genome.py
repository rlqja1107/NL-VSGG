import torch
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from cv2 import imread
import numpy as np
import pickle
import os
from copy import deepcopy
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


def file_download(file_name):
    os.system(f"wget https://huggingface.co/datasets/kb-kim/NL-VSGG/resolve/main/{file_name}")
    os.system(f"mv {file_name} datasets/AG/")

class AG_Train(Dataset):

    def __init__(self, data_path=None, pseudo_label_path = 'datasets/AG/final_ag_data_w_neg.pkl', save_path=None, logger=None):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes4gt = deepcopy(self.object_classes)
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        self.object_classes4gt[9] = 'cabinet'
        self.object_classes4gt[11] = 'glass'
        self.object_classes4gt[23] = 'paper'
        self.object_classes4gt[24] = 'phone'
        self.object_classes4gt[31] = 'sofa'



        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes4gt = deepcopy(self.relationship_classes)
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.relationship_classes4gt[0] = 'looking at'
        self.relationship_classes4gt[1] = 'not looking at'
        self.relationship_classes4gt[5] = 'in front of'
        self.relationship_classes4gt[7] = 'on the side of'
        self.relationship_classes4gt[10] = 'covered by'
        self.relationship_classes4gt[11] = 'drinking from'
        self.relationship_classes4gt[13] = 'have it on the back'
        self.relationship_classes4gt[15] = 'leaning on'
        self.relationship_classes4gt[16] = 'lying on'
        self.relationship_classes4gt[17] = 'not contacting'
        self.relationship_classes4gt[18] = 'other relationship'
        self.relationship_classes4gt[19] = 'sitting on'
        self.relationship_classes4gt[20] = 'standing on'
        self.relationship_classes4gt[25] = 'writing on'


        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]


        print('-------loading annotations---------slowly-----------')

        if not os.path.isfile("datasets/AG/ag_img_info_train.pkl"):
            file_download("ag_img_info_train.pkl")
        with open("datasets/AG/ag_img_info_train.pkl", 'rb') as f:
            img_info = pickle.load(f)

        if not os.path.isfile("datasets/AG/triplets_LLM4SGG.pkl"):
            file_download("triplets_LLM4SGG.pkl")
        with open(f"datasets/AG/triplets_LLM4SGG.pkl", 'rb') as f: # This file contains frame list
            frame_list_info = pickle.load(f)

        if pseudo_label_path != '':
            if not os.path.isfile("datasets/AG/final_ag_data_w_neg.pkl"):
                file_download("final_ag_data_w_neg.pkl")
            with open(pseudo_label_path, 'rb') as f:
                final_ag_data_w_neg = pickle.load(f)
            logger.info(f"Load Pre-processed Pseudo labeled Dataset from '{pseudo_label_path}'")
        else:
            assert "No valid data path!"
        
        self.video_list = []
        self.gt_annotations = []
        triplet_cnt = 0
        total_frame = 0
        action_count = {i:0 for i in self.relationship_classes4gt}
        self.img_info = []

        for video_index, wk_ag_data in final_ag_data_w_neg.items():
            video_list = []
            if video_index not in frame_list_info: continue
            for frame_id in frame_list_info[video_index]['frame_list']:
                video_list.append(f"{video_index}/{frame_id}")
            self.gt_annotations.append(wk_ag_data)

            self.img_info.append(img_info[video_index])
            self.video_list.append(video_list)
            total_frame += len(video_list)
            for frame_info in wk_ag_data:
                for triplets in frame_info:
                    if 'class' not in triplets: continue
                    for a in triplets['attention_relationship']:
                        a = int(a.item())
                        triplet_cnt += 1
                        action_count[self.relationship_classes4gt[a]] += 1
                    for a in triplets['spatial_relationship']:
                        a = int(a.item())
                        action_count[self.relationship_classes4gt[a+3]] += 1
                        triplet_cnt += 1
                    for a in triplets['contacting_relationship']:
                        a = int(a.item())
                        action_count[self.relationship_classes4gt[a+9]] += 1                        
                        triplet_cnt += 1

        
        logger.info('x'*60)
        logger.info(f"The number of total frame is {total_frame}.")
        logger.info(f"The number of valid tripelt is {triplet_cnt}")
        logger.info('x' * 60)


        # Plot
        action_count = dict(sorted(action_count.items(), key=lambda d:d[1], reverse=True))
        plt.figure(figsize=(10,5))
        x_axis = np.arange(len(action_count))
        y = list(action_count.values())
        x = list(action_count.keys())
        plt.bar(x_axis, y, color='black', alpha=0.5)
        plt.xticks(x_axis, x, rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(f"{save_path}/action_dist.png", bbox_inches='tight')
            
    def __getitem__(self, index):

        img_tensor = None; im_info = self.img_info[index]; gt_boxes = None; num_boxes = None
        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)



class AG_Test(Dataset):

    def __init__(self, mode, logger, data_path=None, filter_nonperson_box_frame=True, filter_small_box=True):

        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.relationship_classes[0] = 'looking_at'
        self.relationship_classes[1] = 'not_looking_at'
        self.relationship_classes[5] = 'in_front_of'
        self.relationship_classes[7] = 'on_the_side_of'
        self.relationship_classes[10] = 'covered_by'
        self.relationship_classes[11] = 'drinking_from'
        self.relationship_classes[13] = 'have_it_on_the_back'
        self.relationship_classes[15] = 'leaning_on'
        self.relationship_classes[16] = 'lying_on'
        self.relationship_classes[17] = 'not_contacting'
        self.relationship_classes[18] = 'other_relationship'
        self.relationship_classes[19] = 'sitting_on'
        self.relationship_classes[20] = 'standing_on'
        self.relationship_classes[25] = 'writing_on'

        self.attention_relationships = self.relationship_classes[0:3]
        self.spatial_relationships = self.relationship_classes[3:9]
        self.contacting_relationships = self.relationship_classes[9:]

        if not os.path.isfile("datasets/AG/ag_img_info_test.pkl"):
            file_download("ag_img_info_test.pkl")
        with open("datasets/AG/ag_img_info_test.pkl", 'rb') as f:
            img_info = pickle.load(f)
        print('-------loading annotations---------slowly-----------')

        if filter_small_box:
            with open(root_path + '/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(root_path +'/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
        else:
            with open(root_path + '/person_bbox.pkl', 'rb') as f:
                person_bbox = pickle.load(f)
            f.close()
            with open(root_path+'/object_bbox_and_relationship.pkl', 'rb') as f:
                object_bbox = pickle.load(f)
            f.close()
        print('--------------------finish!-------------------------')

        # collect valid frames
        video_dict = {}
        for i in person_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('/')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.img_info = []
        self.gt_annotations = []
        self.non_gt_human_nums = 0
        self.non_heatmap_nums = 0
        self.non_person_video = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        '''
        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
        '''
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                if filter_nonperson_box_frame:
                    if person_bbox[j]['bbox'].shape[0] == 0:
                        self.non_gt_human_nums += 1
                        continue
                    else:
                        video.append(j)
                        self.valid_nums += 1


                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
                        k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
                        k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
                        k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            if len(video) > 2:
                self.video_list.append(video)
                self.video_size.append(person_bbox[j]['bbox_size'])
                self.img_info.append(img_info[i])
                self.gt_annotations.append(gt_annotation_video)
            elif len(video) == 1:
                self.one_frame_video += 1
            else:
                self.non_person_video += 1

        logger.info('x'*60)
        if filter_nonperson_box_frame:
            logger.info('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            logger.info('{} videos are invalid (no person), remove them'.format(self.non_person_video))
            logger.info('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
            logger.info('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
        else:
            logger.info('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
            logger.info('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
        logger.info('x' * 60)

    def __getitem__(self, index):

        im_info = self.img_info[index]
        return im_info, index

    def __len__(self):
        return len(self.video_list)


#class AG(Dataset):

#    def __init__(self, mode, datasize, data_path=None, ws_object_bbox_path=None, remove_one_frame_video=True, \
#    filter_nonperson_box_frame=True, filter_small_box=False, need_relation=True, output_log=False, logger=None):

#        root_path = data_path
#        self.frames_path = os.path.join(root_path, 'frames/')

#        # collect the object classes
#        self.object_classes = ['__background__']
#        with open(os.path.join(root_path, 'object_classes.txt'), 'r') as f:
#            for line in f.readlines():
#                line = line.strip('\n')
#                self.object_classes.append(line)
#        f.close()
#        self.object_classes[9] = 'closet/cabinet'
#        self.object_classes[11] = 'cup/glass/bottle'
#        self.object_classes[23] = 'paper/notebook'
#        self.object_classes[24] = 'phone/camera'
#        self.object_classes[31] = 'sofa/couch'

#        # collect relationship classes
#        self.relationship_classes = []
#        with open(os.path.join(root_path, 'relationship_classes.txt'), 'r') as f:
#            for line in f.readlines():
#                line = line.strip('\n')
#                self.relationship_classes.append(line)
#        f.close()
#        self.relationship_classes[0] = 'looking_at'
#        self.relationship_classes[1] = 'not_looking_at'
#        self.relationship_classes[5] = 'in_front_of'
#        self.relationship_classes[7] = 'on_the_side_of'
#        self.relationship_classes[10] = 'covered_by'
#        self.relationship_classes[11] = 'drinking_from'
#        self.relationship_classes[13] = 'have_it_on_the_back'
#        self.relationship_classes[15] = 'leaning_on'
#        self.relationship_classes[16] = 'lying_on'
#        self.relationship_classes[17] = 'not_contacting'
#        self.relationship_classes[18] = 'other_relationship'
#        self.relationship_classes[19] = 'sitting_on'
#        self.relationship_classes[20] = 'standing_on'
#        self.relationship_classes[25] = 'writing_on'

#        self.attention_relationships = self.relationship_classes[0:3]
#        self.spatial_relationships = self.relationship_classes[3:9]
#        self.contacting_relationships = self.relationship_classes[9:]


#        print('-------loading annotations---------slowly-----------')

#        # always use object_bbox_and_relationship_filtersmall.pkl
#        if ws_object_bbox_path is None:
#            print('loading ' + 'gt_annotation_thres02.pkl')
#            with open(os.path.join(root_path, 'gt_annotation_thres02.pkl'), 'rb') as f:
#                person_object_bbox_relationship = pickle.load(f)
#        else:
#            print('loading ' + ws_object_bbox_path)
#            with open(os.path.join(root_path, ws_object_bbox_path), 'rb') as f:
#                person_object_bbox_relationship = pickle.load(f)
#        f.close()

#        #if filter_small_box:
#        #    with open(root_path + '/person_bbox.pkl', 'rb') as f:
#        #        person_bbox = pickle.load(f)
#        #    f.close()
#        #    with open(root_path+'/object_bbox_and_relationship_filtersmall.pkl', 'rb') as f:
#        #        object_bbox = pickle.load(f)
#        #else:
#        #    with open(root_path + '/person_bbox.pkl', 'rb') as f:
#        #        person_bbox = pickle.load(f)
#        #    f.close()
#        #    with open(root_path+'/object_bbox_and_relationship.pkl', 'rb') as f:
#        #        object_bbox = pickle.load(f)
#        #    f.close()
#        #print('--------------------finish!-------------------------')

#        person_bbox = {}
#        object_bbox = {}
#        for frame_name in person_object_bbox_relationship.keys():
#            person_bbox[frame_name] = person_object_bbox_relationship[frame_name]['person_info']
#            object_bbox[frame_name] = person_object_bbox_relationship[frame_name]['object_info']

#        print('--------------------finish!-------------------------')

#        if datasize == 'mini':
#            small_person = {}
#            small_object = {}
#            for i in list(person_bbox.keys())[:1000]: # 1000, 80000
#                small_person[i] = person_bbox[i]
#                small_object[i] = object_bbox[i]
#            person_bbox = small_person
#            object_bbox = small_object


#        # collect valid frames
#        video_dict = {}
#        for i in person_bbox.keys():
#            if len(object_bbox[i]) > 0 and object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
#                frame_valid = False
#                for j in object_bbox[i]: # the frame is valid if there is visible bbox
#                    if j['visible']:
#                        if need_relation:
#                            if j['attention_relationship'] is not None and j['spatial_relationship'] is not None and j['contacting_relationship'] is not None and \
#                                j['attention_relationship'] != [] and j['spatial_relationship'] != [] and j['contacting_relationship'] != []:
#                                # also, at least one object has all relationship
#                                frame_valid = True
#                        else:
#                            frame_valid = True
#                if frame_valid:
#                    video_name, frame_num = i.split('/')
#                    if video_name in video_dict.keys():
#                        video_dict[video_name].append(i)
#                    else:
#                        video_dict[video_name] = [i]

#        self.video_list = []
#        self.video_size = [] # (w,h)
#        self.gt_annotations = []
#        self.non_gt_human_nums = 0
#        self.non_heatmap_nums = 0
#        self.non_person_video = 0
#        self.one_frame_video = 0
#        self.valid_nums = 0
#        self.object_cnt = {i : 0 for i in self.object_classes}
#        self.relationship_cnt = {i : 0 for i in self.relationship_classes}

#        '''
#        filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
#        filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
#        '''
#        for i in video_dict.keys():
#            video = []
#            gt_annotation_video = []
#            for j in video_dict[i]:
#                if filter_nonperson_box_frame:
#                    if person_bbox[j]['bbox'].shape[0] == 0:
#                        self.non_gt_human_nums += 1
#                        continue
#                    else:
#                        video.append(j)
#                        self.valid_nums += 1


#                gt_annotation_frame = [{'person_bbox': person_bbox[j]['bbox']}]
#                # each frames's objects and human
#                for k in object_bbox[j]:
#                    if k['visible']:
#                    # if k['bbox'] != None and k['visible']:
#                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
#                        if need_relation == False:
#                            self.object_cnt[k['class']] += 1

#                            if k['attention_relationship'] is not None:
#                                for ar in k['attention_relationship']:
#                                    self.relationship_cnt[ar] += 1
#                            if k['spatial_relationship'] is not None:
#                                for sr in k['spatial_relationship']:
#                                    self.relationship_cnt[sr] += 1
#                            if k['contacting_relationship'] is not None:
#                                for cr in k['contacting_relationship']:
#                                    self.relationship_cnt[cr] += 1
                                
#                            k['class'] = self.object_classes.index(k['class'])
#                            '''
#                            if mode == 'test':
#                                k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
#                            elif mode == 'train':
#                                k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
#                            '''
#                            if k['attention_relationship'] is not None:
#                                k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
#                            if k['spatial_relationship'] is not None:
#                                k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
#                            if k['contacting_relationship'] is not None:
#                                k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
#                            gt_annotation_frame.append(k)

#                        elif need_relation and k['attention_relationship'] is not None and k['spatial_relationship'] is not None and k['contacting_relationship'] is not None and \
#                            k['attention_relationship'] != [] and k['spatial_relationship'] != [] and k['contacting_relationship'] != []:

#                            self.object_cnt[k['class']] += 1
#                            for ar in k['attention_relationship']:
#                                self.relationship_cnt[ar] += 1
#                            for sr in k['spatial_relationship']:
#                                self.relationship_cnt[sr] += 1
#                            for cr in k['contacting_relationship']:
#                                self.relationship_cnt[cr] += 1
                                
#                            k['class'] = self.object_classes.index(k['class'])
#                            '''
#                            if mode == 'test':
#                                k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
#                            elif mode == 'train':
#                                k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][0]+k['bbox'][2], k['bbox'][1]+k['bbox'][3]]) # from xywh to xyxy
#                            '''
#                            k['attention_relationship'] = torch.tensor([self.attention_relationships.index(r) for r in k['attention_relationship']], dtype=torch.long)
#                            k['spatial_relationship'] = torch.tensor([self.spatial_relationships.index(r) for r in k['spatial_relationship']], dtype=torch.long)
#                            k['contacting_relationship'] = torch.tensor([self.contacting_relationships.index(r) for r in k['contacting_relationship']], dtype=torch.long)
#                            gt_annotation_frame.append(k)

#                gt_annotation_video.append(gt_annotation_frame)

#            if len(video) > 2:
#                self.video_list.append(video)   # e.g. '001YG.mp4/000089.png'
#                self.video_size.append(person_bbox[j]['bbox_size'])
#                self.gt_annotations.append(gt_annotation_video)
                
#            elif len(video) == 1 or len(video) == 2:
#                if remove_one_frame_video:
#                    self.one_frame_video += 1
#                else:
#                    self.video_list.append(video)   # e.g. '001YG.mp4/000089.png'
#                    self.video_size.append(person_bbox[j]['bbox_size'])
#                    self.gt_annotations.append(gt_annotation_video)
#            else:
#                self.non_person_video += 1

#        logger.info('x'*60)
#        if filter_nonperson_box_frame: # True
#            logger.info('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
#            logger.info('{} videos are invalid (no person), remove them'.format(self.non_person_video))
#            logger.info('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
#            logger.info('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
#        else:
#            print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
#            print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
#            print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(self.non_heatmap_nums))
#        logger.info('x' * 60)

#        bbox_cnt = 0
#        people_cnt = 0
#        for gt_annotation_video in self.gt_annotations:
#            for gt_annotation_frame in gt_annotation_video:
#                bbox_cnt += len(gt_annotation_frame)
#                people_cnt += 1
#        logger.info('There are {} bboxes, including {} people and {} objects'.format(bbox_cnt, people_cnt, bbox_cnt-people_cnt))

#        logger.info('x' * 60)
#        if output_log:
#            logger.info('object frequency: ' + str(self.object_cnt))
#            logger.info('relationship frequency: ' + str(self.relationship_cnt))

#        not_appear_object = []
#        not_appear_relationship = []
#        object_count = 0
#        relationship_count = 0
#        for name, cnt in self.object_cnt.items():
#            object_count += cnt
#            if cnt == 0:
#                not_appear_object.append(name)
#        for name, cnt in self.relationship_cnt.items():
#            relationship_count += cnt
#            if cnt == 0:
#                not_appear_relationship.append(name)
        
#        logger.info('There are {} objects and {} relationships'.format(object_count, relationship_count))
        
#        if output_log:
#            logger.info('object ' + str(not_appear_object) + ' not appear')
#            logger.info('relationship ' + str(not_appear_relationship) + ' not appear')

#        logger.info('x' * 60)

#    def __getitem__(self, index):

#        frame_names = self.video_list[index]
#        processed_ims = []
#        im_scales = []

#        for idx, name in enumerate(frame_names):
#            im = imread(os.path.join(self.frames_path, name)) # channel h,w,3
#            # im = im[:, :, ::-1] # rgb -> bgr
#            im, im_scale = prep_im_for_blob(im, [[[102.9801, 115.9465, 122.7717]]], 600, 1000) #cfg.PIXEL_MEANS, target_size, cfg.TRAIN.MAX_SIZE
#            im_scales.append(im_scale)
#            processed_ims.append(im)

#        blob = im_list_to_blob(processed_ims)
#        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
#        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
#        img_tensor = torch.from_numpy(blob)
#        img_tensor = img_tensor.permute(0, 3, 1, 2)

#        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
#        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

#        """
#        img_tensor: 网络输入的torch
#        im_info: blob.shape[0]个[blob.shape[1], blob.shape[2], im_scales[0]]组成的二维数组
#        gt_boxes: 空torch(num*1*5)
#        num_boxes: 空torch(num)
#        index: 等于输入的index
#        """
#        return img_tensor, im_info, gt_boxes, num_boxes, index

#    def __len__(self):
#        return len(self.video_list)


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]
