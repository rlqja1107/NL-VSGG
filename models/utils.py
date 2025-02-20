import os
import json
import torch
from PIL import Image
from typing import List
import models.dataset as T
from models.misc import nested_tensor_from_tensor_list, get_world_size
import torch.distributed as dist
from bisect import bisect_right

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target = None):
        for t in self.transforms:
            image, target = t(image, target = target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
    
    
def make_hico_transforms(image_set):

    normalize = Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if image_set == 'val':
        return Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])
    raise ValueError(f'unknown {image_set}')


def split_path_list(img_root_path, batch_size):
    # Split the list by batch_size
    path_list = os.listdir(img_root_path)
    img_path_list = []
    for path in path_list:
        if os.path.isfile(img_root_path + path):
            img_path_list.append(path)

    batch_img_path = []
    temp_img_path = []
    for img_path in img_path_list:
        temp_img_path.append(img_root_path + img_path)
        if len(temp_img_path) == batch_size:
            batch_img_path.append(temp_img_path)
            temp_img_path = []
    if len(temp_img_path) > 0:
        batch_img_path.append(temp_img_path)

    return batch_img_path




def load_image(transform, file_path_list, device):
    raw_image_list = []
    size_list = []
    for file_path in file_path_list:
        raw_image = Image.open(file_path).convert('RGB')
        w, h = raw_image.size
        raw_image_list.append(raw_image)  
        size_list.append(torch.as_tensor([int(h), int(w)]).to(device))

    image = [transform(raw_image)[0].to(device) for raw_image in raw_image_list]
    image = nested_tensor_from_tensor_list(image)
    size = torch.stack(size_list, dim = 0)
    return image, size




def load_hico_verb_txt(file_path = 'classes_txt/ai_verb_names.txt') -> List[list]:
    '''
    Output like [['adjust'], ['board'], ['brush', 'with'], ['buy']]
    '''
    verb_names = []
    for line in open(file_path,'r'):
        # verb_names.append(line.strip().split(' ')[-1])
        verb_names.append(' '.join(line.strip().split(' ')[-1].split('_')))
    return verb_names

def load_hico_object_txt(file_path = 'classes_txt/ai_object_names.txt') -> List[list]:
    '''
    Output like [['train'], ['boat'], ['traffic', 'light'], ['fire', 'hydrant']]
    '''
    object_names = []
    with open(file_path, 'r') as f:
        object_names = json.load(f)
    object_list = list(object_names.keys())
    return object_list


def adjust_learning_rate(
    optimizer,
    epoch: int,
    curr_step: int,
    num_training_steps: int,
    args,
):
    """Adjust the lr according to the schedule.

    Args:
        Optimizer: torch optimizer to update.
        epoch(int): number of the current epoch.
        curr_step(int): number of optimization step taken so far.
        num_training_step(int): total number of optimization steps.
        args: additional training dependent args:
              - lr_drop(int): number of epochs before dropping the learning rate.
              - fraction_warmup_steps(float) fraction of steps over which the lr will be increased to its peak.
              - lr(float): base learning rate
              - lr_backbone(float): learning rate of the backbone
              - text_encoder_backbone(float): learning rate of the text encoder
              - schedule(str): the requested learning rate schedule:
                   "step": all lrs divided by 10 after lr_drop epochs
                   "multistep": divided by 2 after lr_drop epochs, then by 2 after every 50 epochs
                   "linear_with_warmup": same as "step" for backbone + transformer, but for the text encoder, linearly
                                         increase for a fraction of the training, then linearly decrease back to 0.
                   "all_linear_with_warmup": same as "linear_with_warmup" for all learning rates involved.

    """
    num_warmup_steps: int = round(args.fraction_warmup_steps * num_training_steps)
    if args.schedule == "step":
        gamma = 0.1 ** (epoch // args.lr_drop)
        text_encoder_gamma = gamma
    elif args.schedule == "multistep":
        milestones = list(range(args.lr_drop, args.nepoch, 50))
        gamma = 0.5 ** bisect_right(milestones, epoch)
        text_encoder_gamma = gamma
    elif args.schedule == "linear_with_warmup":
        gamma = 0.1 ** (epoch // args.lr_drop)
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step) / float(max(1, num_training_steps - num_warmup_steps)),
            )
    elif args.schedule == "all_linear_with_warmup":
        if curr_step < num_warmup_steps:
            text_encoder_gamma = float(curr_step) / float(max(1, num_warmup_steps))
        else:
            text_encoder_gamma = max(
                0.0,
                float(num_training_steps - curr_step) / float(max(1, num_training_steps - num_warmup_steps)),
            )
        gamma = text_encoder_gamma
    else:
        raise NotImplementedError

    base_lrs = [args.lr, args.lr_backbone, args.text_encoder_lr]
    gammas = [gamma, gamma, text_encoder_gamma]
    #assert len(optimizer.param_groups)-1 == len(base_lrs) # First one is STTran
    if len(optimizer.param_groups) > 3:
        for param_group, lr, gamma_group in zip(optimizer.param_groups[1:], base_lrs, gammas):
            param_group["lr"] = lr * gamma_group
    else:
        for param_group, lr, gamma_group in zip(optimizer.param_groups, base_lrs, gammas):
            param_group["lr"] = lr * gamma_group        



def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            # print(k)
            # print(input_dict[k].device)
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


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