import torch

def check_valid_iter(object_label, contact_distribution, spatial_distribution, attention_distribution, loss):
    skip = False
    if len(object_label) == 0:
        skip = True
    if torch.isnan(contact_distribution).sum() > 0 or torch.isnan(spatial_distribution).sum() > 0 or torch.isnan(attention_distribution).sum() > 0:
        skip = True
    
    if torch.isnan(loss):
        skip = True
    return skip