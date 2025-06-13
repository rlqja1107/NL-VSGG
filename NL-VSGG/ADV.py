import sys
import torch
import pickle
import random
import itertools
import numpy as np
import pandas as pd
sys.path.append('.')
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

# You need to modify the path in `default_bpe` function to 'DAC/vl_checklist_annot_data/bpe_simple_vocab_16e6.txt.gz'
from DAC.src.open_clip import tokenize
from DAC.src.open_clip import create_model_and_transforms

# Fix seed
torch.manual_seed(1000)
torch.cuda.manual_seed(1000)
torch.cuda.manual_seed_all(1000)  # if you are using multi-GPU.
np.random.seed(1000)  # Numpy module.
random.seed(1000)  # Python random module.
torch.manual_seed(1000)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_num_threads(4)

BETA = 4
model, _, image_preprocess = create_model_and_transforms(
    "ViT-B/32",
    "openai",
    precision='amp',
    device='cpu',
    jit=False,
    force_quick_gelu=False,
    pretrained_image=False,
    image_mean=None,
    image_std=None,
    lora=4,
    freeze_img=False,
    kqv_lora=False,
)

model.eval()

# Load pre-trained DAC model
checkpoint = torch.load("DAC/LLM_cp.pt", map_location='cpu')
sd = checkpoint['state_dict']
if next(iter(sd.items()))[0].startswith("module"):
    sd = {k[len("module.") :]: v for k, v in sd.items()}
model.load_state_dict(sd)

with open("datasets/AG/triplets_LLM4SGG.pkl", 'rb') as f:
    split_sentence = pickle.load(f)

# Load necessary files
ag_caption = pd.read_csv("datasets/AG/Charades_vu17_train.csv")
with open('datasets/AG/ag_train_id.pkl', 'rb') as f:
    video_frame_dict = pickle.load(f)
caption_dict = {}

# Video Caption
for k, v in ag_caption.iterrows():
    video_id = v['id'] + ".mp4"
    caption_dict[video_id] = v['descriptions']
    

video_index_list = list(split_sentence.keys())

def find_consecutive_numbers(lst):
    consecutive_groups = []
    current_group = []

    for num in sorted(lst):
        if not current_group or num == current_group[-1] + 1:
            current_group.append(num)
        else:
            consecutive_groups.append(current_group)
            current_group = [num]

    # Append the last group
    consecutive_groups.append(current_group)

    # Find the longest consecutive group
    longest_group = max(consecutive_groups, key=len, default=[])

    return longest_group

video_key = []
preprocessed_data_dict = {}

for video_index in tqdm(video_index_list):
    preprocessed_data_dict[video_index] = {}
    video_key.append(video_index)
    preprocessed_data_dict[video_index]['mapped_frame'] = [[] for _ in range(len(list(itertools.chain.from_iterable(split_sentence[video_index]['split_sentence']))))]
    ith = 0
    for video_captions in split_sentence[video_index]['split_sentence']:
        if len(video_captions) == 0:
            continue
        torch.cuda.empty_cache()
        
        # Text Feature
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_tokenize = torch.cat([tokenize(c) for c in video_captions])
            text_features = model.encode_text(text_tokenize)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        video_tensor = []
        for f in video_frame_dict[video_index]:
            video_path = f'datasets/frames/{video_index}/{f}'
            video_path = f'/home/public/Datasets/CV/video/frames/{video_index}/{f}'
            image = image_preprocess(Image.open(video_path)).unsqueeze(0)
            video_tensor.append(image)
            
        # Video Feature
        with torch.no_grad(), torch.cuda.amp.autocast():
            visual_features = model.encode_image(torch.cat(video_tensor))
            visual_features /= visual_features.norm(dim=-1, keepdim=True)

        k = int(len(video_tensor) / BETA)
        if k <= 1:
            k = 2

        if len(video_tensor) > 5:
            cluster_result = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(visual_features.numpy())
            labels = cluster_result.labels_
            cluster_scores = torch.FloatTensor(100.0 * cluster_result.cluster_centers_ @ text_features.numpy().T).softmax(dim=0)
            
            cluster_scores = cluster_scores.numpy()
            sort_scores = np.sort(cluster_scores, axis=0)
            scores_diff = np.diff(sort_scores, axis=0)
            sort_idx = scores_diff.argmax(axis=0) + 1
            nth_sentence = (sort_scores[sort_idx, np.arange(sort_scores.shape[1])]<= cluster_scores).nonzero()[1]
            candidate = (sort_scores[sort_idx, np.arange(sort_scores.shape[1])]<= cluster_scores).nonzero()[0]
            cluster_score_sort_idx = [[] for _ in range(cluster_scores.shape[1])]
            for nth, c in zip(nth_sentence, candidate):
                cluster_score_sort_idx[nth].append(c)
                
            unrealistic_condition = 0
            for nth_sentence, c_s in enumerate(cluster_score_sort_idx):
                c_s = np.array(c_s)[:2]
                if len(c_s) > 2:
                    overlap_idx = (c_s[None,...] == labels[...,None]).nonzero()[1]
                    unique_elements, unique_indices = np.unique(overlap_idx, return_index=True)
                    sorted_indices = np.sort(unique_indices)
                    unique_elements_in_sequence = overlap_idx[sorted_indices]
                    c_s = c_s[unique_elements_in_sequence[:2]]
                matched_frame = (labels[None,...] == c_s[...,None]).any(0).nonzero()[0]
                valid_frame_idx = matched_frame >= unrealistic_condition
                matched_frame = matched_frame[valid_frame_idx]
                matched_frame = find_consecutive_numbers(matched_frame)
                if len(matched_frame) == 0:
                    ith += 1
                    continue
                unrealistic_condition = matched_frame[0]
                for m in matched_frame:
                    m = np.arange(m, m + 1) if len(video_tensor) > m + 1 else np.arange(m, len(video_tensor))
                    preprocessed_data_dict[video_index]['mapped_frame'][ith].extend(m)
                ith += 1
    
        else:
            # Handle short length video
            time_stamp = 1
            cluster_scores = torch.FloatTensor(100.0 * visual_features.numpy() @ text_features.numpy().T).softmax(dim=0)

            cluster_scores = cluster_scores.numpy()
            sort_scores = np.sort(cluster_scores, axis=0)
            scores_diff = np.diff(sort_scores, axis=0)
            sort_idx = scores_diff.argmax(axis=0) + 1
            nth_sentence = (sort_scores[sort_idx, np.arange(sort_scores.shape[1])]<= cluster_scores).nonzero()[1]
            candidate = (sort_scores[sort_idx, np.arange(sort_scores.shape[1])]<= cluster_scores).nonzero()[0]
            cluster_score_sort_idx = [[] for _ in range(cluster_scores.shape[1])]
            for nth, c in zip(nth_sentence, candidate):
                cluster_score_sort_idx[nth].append(c)
                
            unrealistic_condition = 0
            
            for c_s in cluster_score_sort_idx:
                matched_frame = np.array(c_s)[:2]
                valid_idx = matched_frame >= unrealistic_condition
                matched_frame = matched_frame[valid_idx]
                matched_frame = find_consecutive_numbers(matched_frame)
                if len(matched_frame) == 0:
                    ith += 1
                    continue
                unrealistic_condition = matched_frame[0]      
                for m in matched_frame:
                    m = np.arange(m, m+1) if len(video_tensor) > m + 1 else np.arange(m, len(video_tensor))
                    preprocessed_data_dict[video_index]['mapped_frame'][ith].extend(m)
                ith += 1


preprocessed_data_dict4save = {}
tri_cnt = 0 
for k in video_key:
    preprocessed_data_dict4save[k] = {}
    preprocessed_data_dict4save[k]['frame_list'] = split_sentence[k]['frame_list']
    preprocessed_data_dict4save[k]['split_sentence'] = list(itertools.chain.from_iterable(split_sentence[k]['split_sentence']))
    parsed_triplets = list(itertools.chain.from_iterable(split_sentence[k]['triplets']))
    new_parsed_triplets = []
    for i, tris in enumerate(parsed_triplets):
        temp_parsed_triplets = []
        for j, tri in enumerate(tris):
            if tri[0] != 'person' or tri[1] == 'unsure':
                continue
            temp_parsed_triplets.append(tri)
        new_parsed_triplets.append(temp_parsed_triplets)
    preprocessed_data_dict4save[k]['triplets'] = new_parsed_triplets
    preprocessed_data_dict4save[k]['mapped_frame'] = preprocessed_data_dict[k]['mapped_frame']


valid_video_cnt = 0
triplet_cnt = 0
error_case = set()
for k, v in preprocessed_data_dict4save.items():
    triplet_list = [[] for _ in range(len(v['frame_list']))]
    if len(v['triplets']) != len(v['mapped_frame']): error_case.add(k)
    
    for triplets, matched_frame_id in zip(v['triplets'], v['mapped_frame']):
        obj_set = set()
        for triplet in triplets:
            for m_f in matched_frame_id:
                if m_f < len(v['frame_list']):
                    triplet_list[m_f].append((triplet[0], triplet[1], triplet[2]))
                else:
                    error_case.add(k)
    for i, tri in enumerate(triplet_list):
        triplet_list[i] =list(set(tri))
        triplet_cnt += len(triplet_list[i])
    preprocessed_data_dict4save[k]['triplets'] = triplet_list
    valid_video_cnt += 1

for i in error_case:
    del preprocessed_data_dict4save[k]

# Save Intermediate Result
with open(f"datasets/AG/semi_final_ag_data.pkl", 'wb') as f:
    pickle.dump(preprocessed_data_dict4save, f)


# Pre-process dataset into format
action_class = ['looking at', 'not looking at', 'unsure', 'above', 'beneath', 'in front of', 'behind', 'on the side of', 'in', 'carrying', 'covered by', 'drinking from', 'eating', 'have it on the back', 'holding', 'leaning on', 'lying on', 'not contacting', 'other relationship', 'sitting on', 'standing on', 'touching', 'twisting', 'wearing', 'wiping', 'writing on']    
obj_class = ['__background__']
with open("datasets/AG/object_classes.txt", 'r') as f:
    for i in f.readlines():
        obj_class.append(i.strip('\n'))
obj_class[9] = 'cabinet'
obj_class[11] = 'cup'
obj_class[23] = 'paper'
obj_class[24] = 'phone'
obj_class[31] = 'sofa'


attention_classes = action_class[:3]
spatial_classes = action_class[3:9]
contacting_classes = action_class[9:]

def add_info_vanilla(rel_class, video_index, nth_frame, obj_index,  person_bbox, object_bbox, final_data_dict):
    attention_list = []; spatial_list = []; contacting_list = []
    if rel_class in attention_classes:
        rel_type = 'attention_relationship'
        rel_idx = attention_classes.index(rel_class)
        attention_list.append(rel_idx)
    elif rel_class in spatial_classes:
        rel_type = 'spatial_relationship'
        rel_idx = spatial_classes.index(rel_class)
        spatial_list.append(rel_idx)
    else:
        rel_type = 'contacting_relationship'
        rel_idx = contacting_classes.index(rel_class)
        contacting_list.append(rel_idx)
        
    if len(final_data_dict[video_index][nth_frame]) == 0:
        final_data_dict[video_index][nth_frame].append({'person_bbox': person_bbox})
    exist = False
    for per_frame_data in final_data_dict[video_index][nth_frame]:
        if 'class' not in per_frame_data: continue
        if per_frame_data['class'] == obj_index:
            per_frame_data[rel_type] = torch.unique(torch.cat([per_frame_data[rel_type], torch.as_tensor([rel_idx])]))
            exist = True
            break
    if not exist:
        final_data_dict[video_index][nth_frame].append({'class': obj_index, 'bbox': object_bbox, 'attention_relationship': torch.as_tensor(attention_list), 'spatial_relationship': torch.as_tensor(spatial_list), 'contacting_relationship': torch.as_tensor(contacting_list)})      


remove_id_set = []
final_data_dict = {}
for video_index, v in preprocessed_data_dict4save.items():
    final_data_dict[video_index] = [[] for _ in range(len(v['frame_list']))]
    tri_cnt = 0
    for nth_frame, triplets in enumerate(v['triplets']):
        for triplet in triplets:
            if triplet[2] == 'glass':
                triplet = list(triplet)
                triplet[2] = 'cup'
            tri_cnt += 1
            add_info_vanilla(triplet[1], video_index, nth_frame, obj_class.index(triplet[2]), np.array([0]*4), np.array([0]*4), final_data_dict)
    if tri_cnt == 0: remove_id_set.append(video_index)
    
for r in remove_id_set:
    del final_data_dict[r]

print(f"Total Valid Video: {valid_video_cnt} / Triplet: {triplet_cnt}")

# Save Final Result
with open(f"datasets/AG/final_ag_data.pkl", 'wb') as f:
    pickle.dump(final_data_dict, f)