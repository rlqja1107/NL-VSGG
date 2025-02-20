import torch


# 构造一个空的entry
def make_null_entry():
    entry = {}
    entry['boxes'] = torch.zeros(0, 5).cuda()
    entry['labels'] = torch.zeros(0).cuda()
    entry['scores'] = torch.zeros(0).cuda()
    entry['distribution'] = torch.zeros(0, 36).cuda()
    entry['im_idx'] = torch.zeros(0).cuda()
    entry['pair_idx'] = torch.zeros(0, 2).cuda()
    entry['features'] = torch.zeros(0, 2048).cuda()
    entry['union_feat'] = torch.zeros(0, 2048, 7, 7).cuda()
    entry['spatial_masks'] = torch.zeros(0, 2, 27, 27).cuda()
    entry['attention_gt'] = torch.tensor([[]]).cuda()
    entry['spatial_gt'] = torch.tensor([[]]).cuda()
    entry['contacting_gt'] = torch.tensor([[]]).cuda()
    return entry


# 构造一个空的pred
def make_null_pred():
    pred = {}
    pred['boxes'] = torch.zeros(0, 5).cuda() 
    pred['scores'] = torch.zeros(0).cuda()
    pred['distribution'] = torch.zeros(0, 36).cuda()
    pred['pred_labels'] = torch.zeros(0).cuda()
    pred['features'] = torch.zeros(0, 2048).cuda()
    pred['fmaps'] = None
    pred['im_info'] = None
    pred['scores'] = torch.zeros(0).cuda()
    pred['pair_idx'] = torch.zeros(0, 2).cuda()
    pred['im_idx'] = torch.zeros(0).cuda()
    pred['human_idx'] = None
    pred['union_feat'] = torch.zeros(0, 2048, 7, 7).cuda()
    pred['union_box'] = torch.zeros(0, 5).cuda() 
    pred['spatial_masks'] = torch.zeros(0, 2, 27, 27).cuda()
    pred['attention_distribution'] = torch.tensor([]).cuda()
    pred['spatial_distribution'] = torch.tensor([]).cuda()
    pred['contacting_distribution'] = torch.tensor([]).cuda()
    return pred