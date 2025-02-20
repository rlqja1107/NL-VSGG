import torch
import math
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from models.misc import accuracy, is_dist_avail_and_initialized, get_world_size




def norm_tensor(tensor):
    norm = torch.norm(tensor, p = 'fro', dim = -1).unsqueeze(dim = -1).expand_as(tensor)
    return tensor/norm


class SetCriterionHOI(nn.Module):
    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, 
                       eos_coef, losses, verb_loss_type, obj_loss_type = 'cross_entropy', temperature = 0.07, 
                       matching_symmetric = True, ParSeDETRHOI = False, subject_class = False, use_no_verb_token = False, 
                       giou_verb_label = False, verb_curing = False, pseudo_verb = False, triplet_filtering = False,
                       naive_obj_smooth = 0, naive_verb_smooth = 0):
        super().__init__()

        assert verb_loss_type in ['weighted_bce', 'focal', 'focal_without_sigmoid', 'focal_bce', 'asymmetric_bce', 'CB_focal_bce','bce', 'cross_modal_matching']
        assert obj_loss_type in ['cross_entropy', 'cross_modal_matching', 'cross_entropy_with_tem', 'cross_entropy_with_tem_focal',
                                 'cross_entropy_symmetric']
        print('verb_loss_type:', verb_loss_type, ';', 'obj_loss_type:', obj_loss_type)
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.verb_loss_type = verb_loss_type
        self.obj_loss_type = obj_loss_type
        self.temperature = temperature
        self.matching_symmetric = matching_symmetric
        self.ParSeDETRHOI = ParSeDETRHOI
        self.subject_class = subject_class
        self.use_no_verb_token = use_no_verb_token
        self.giou_verb_label = giou_verb_label
        if self.giou_verb_label:
            assert verb_loss_type == 'focal' and obj_loss_type == 'cross_entropy'
        self.pseudo_verb = pseudo_verb
        self.verb_curing = verb_curing
        if self.giou_verb_label:
            assert verb_loss_type == 'focal'
        if self.verb_curing:
            assert verb_loss_type == 'focal'
        self.triplet_filtering = triplet_filtering
        if self.triplet_filtering:
            assert self.subject_class # To make sure that we are using VG with subject classes
        self.naive_verb_smooth = naive_verb_smooth
        self.naive_obj_smooth = naive_obj_smooth
        assert ((self.naive_verb_smooth > 0) and self.giou_verb_label) is not True
        print('Use naive_obj_smooth?', self.naive_obj_smooth)
        print('Use naive_verb_smooth?', self.naive_verb_smooth)
        print('Use pseudo_verb?', self.pseudo_verb)
        print('Use verb_curing?', self.verb_curing)
        print('Use triplet_filtering?', self.triplet_filtering)

        # For CB focal
        samples = np.load('datasets/priors/hico_verb_samples.npz')['matrices']
        samples = torch.tensor(samples).float()
        self.register_buffer('samples', samples)
        self.img_num_hico = 37536
        self.img_num_vcoco = 5400
        self.query_num = 100
        self.register_buffer('bce_weight', self.BCE_weight())
    
    def BCE_weight(self,):
        total_num = self.img_num_hico * self.query_num
        pos_verb_samples = self.samples
        neg_verb_samples = total_num - pos_verb_samples
        pos_verb_w = torch.ones(self.samples.shape)
        neg_verb_w = torch.sqrt(pos_verb_samples) / torch.sqrt(neg_verb_samples)
        return torch.stack((pos_verb_w, neg_verb_w), dim = 1)

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        if self.obj_loss_type in ['cross_entropy', 'cross_entropy_with_tem', 'cross_entropy_with_tem_focal', \
                                  'cross_entropy_symmetric']:
            # This means that we are performing the pretraining.
            if self.subject_class:
                ### Calculate loss for objects
                if "with_tem" in self.obj_loss_type:
                    src_logits_obj = outputs['pred_obj_logits']/self.temperature
                else:
                    src_logits_obj = outputs['pred_obj_logits']
                obj_weight = torch.ones(src_logits_obj.shape[-1], device = src_logits_obj.device)
                obj_weight[-1] = self.eos_coef
                idx = self._get_src_permutation_idx(indices)
                # idx: a tuple (batch_idx, src_idx)
                target_classes_o_obj = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits_obj.shape[:2], src_logits_obj.shape[-1] - 1,
                                            dtype=torch.int64, device=src_logits_obj.device)
                # target_classes: init with a tensor of size src_logits_obj.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                if target_classes_o_obj.shape[0] >= 0:
                    target_classes[idx] = target_classes_o_obj
                # fill the target_classes with the gt object classes
                if "focal" in self.obj_loss_type:
                    gamma = 2
                    # alpha = 0.5 # It won't work because every term is multiplied by an alpha.
                    logprobs = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, reduction = 'none')
                    logprobs_weights = obj_weight[target_classes]
                    pt = torch.exp(-logprobs)
                    focal_loss = (1 - pt)**gamma * logprobs * logprobs_weights
                    loss_obj_ce = focal_loss.mean()
                elif "symmetric" in self.obj_loss_type:
                    loss_obj_ce = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, obj_weight)
                    # RCE
                    pred = F.softmax(src_logits_obj, dim = -1)
                    pred = torch.clamp(pred, min = 1e-7, max = 1.0)
                    label_one_hot = torch.nn.functional.one_hot(target_classes, src_logits_obj.shape[-1]).float().to(src_logits_obj.device)
                    label_one_hot = torch.clamp(label_one_hot, min = 1e-4, max = 1.0)
                    loss_obj_rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim = 1))

                    # Loss
                    # loss_obj_ce = 0.1 * loss_obj_ce + 1.0 * loss_obj_rce.mean()
                    loss_obj_ce = 6.0 * loss_obj_ce + 0.1 * loss_obj_rce.mean()

                else:

                    if self.naive_obj_smooth > 0:
                        target_logits_obj = F.one_hot(target_classes, src_logits_obj.shape[-1]).float().to(src_logits_obj.device)
                        target_logits_obj = target_logits_obj * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits_obj.shape[-1]) + \
                                            (1 - target_logits_obj) * self.naive_obj_smooth/src_logits_obj.shape[-1]
                        # print(target_logits_obj.max(-1)[1], target_logits_obj.sum())
                        loss_obj_ce = - (F.log_softmax(src_logits_obj, dim = -1) * target_logits_obj * obj_weight).sum(dim = -1)
                        loss_obj_ce = loss_obj_ce.sum() / obj_weight[target_classes].sum()
                        # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                    else:
                        loss_obj_ce = F.cross_entropy(src_logits_obj.transpose(1, 2), target_classes, obj_weight)
                

                ### Calculate loss for subjects
                if "with_tem" in self.obj_loss_type:
                    src_logits_sub = outputs['pred_sub_logits']/self.temperature
                else:
                    src_logits_sub = outputs['pred_sub_logits']
                sub_weight = torch.ones(src_logits_sub.shape[-1], device = src_logits_sub.device)
                sub_weight[-1] = self.eos_coef
                target_classes_o_sub = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits_sub.shape[:2], src_logits_sub.shape[-1] - 1,
                                            dtype=torch.int64, device=src_logits_sub.device)
                if target_classes_o_sub.shape[0] >= 0:
                    target_classes[idx] = target_classes_o_sub
                
                if "focal" in self.obj_loss_type:
                    gamma = 2
                    # alpha = 0.5 # It won't work because every term is multiplied by an alpha.
                    logprobs = F.cross_entropy(src_logits_sub.transpose(1, 2), target_classes, reduction = 'none')
                    logprobs_weights = sub_weight[target_classes]
                    pt = torch.exp(-logprobs)
                    focal_loss = (1 - pt)**gamma * logprobs * logprobs_weights
                    loss_sub_ce = focal_loss.mean()
                else:

                    if self.naive_obj_smooth > 0:
                        target_logits_sub = F.one_hot(target_classes, src_logits_sub.shape[-1]).float().to(src_logits_sub.device)
                        target_logits_sub = target_logits_sub * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits_sub.shape[-1]) + \
                                            (1 - target_logits_sub) * self.naive_obj_smooth/src_logits_sub.shape[-1]
                        # print(target_logits_sub.max(-1)[0], target_logits_sub.sum())
                        loss_sub_ce = - (F.log_softmax(src_logits_sub, dim = -1) * target_logits_sub * sub_weight).sum(dim = -1)
                        loss_sub_ce = loss_sub_ce.sum() / sub_weight[target_classes].sum() 
                        # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                    else:
                        loss_sub_ce = F.cross_entropy(src_logits_sub.transpose(1, 2), target_classes, sub_weight)
                
                losses = {'loss_obj_ce': loss_obj_ce + loss_sub_ce}
                # losses = {'loss_obj_ce': torch.tensor(0., device = src_logits_sub.device)}
                if log:
                    losses['obj_class_error'] = 100 - accuracy(src_logits_obj[idx], target_classes_o_obj)[0]
                    losses['sub_class_error'] = 100 - accuracy(src_logits_sub[idx], target_classes_o_sub)[0]
                return losses

            else:
                # hack implementation about 'cross_entropy_with_tem' and 'cross_entropy_with_tem_focal'
                assert self.obj_loss_type in ['cross_entropy', 'cross_entropy_with_tem']
                assert 'pred_obj_logits' in outputs
                if "with_tem" in self.obj_loss_type:
                    src_logits = outputs['pred_obj_logits'] / self.temperature
                else:
                    src_logits = outputs['pred_obj_logits']

                idx = self._get_src_permutation_idx(indices)
                # idx: a tuple (batch_idx, src_idx)
                target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(src_logits.shape[:2], src_logits.shape[-1] - 1, # self.num_obj_classes,
                                            dtype=torch.int64, device=src_logits.device)
                # target_classes: init with a tensor of size src_logits.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                target_classes[idx] = target_classes_o
                # fill the target_classes with the gt object classes

                if self.naive_obj_smooth > 0:
                    target_logits = F.one_hot(target_classes, src_logits.shape[-1]).float().to(src_logits.device)
                    target_logits = target_logits * (1 - self.naive_obj_smooth + self.naive_obj_smooth/src_logits.shape[-1]) + \
                                        (1 - target_logits) * self.naive_obj_smooth/src_logits.shape[-1]
                    loss_obj_ce = - (F.log_softmax(src_logits, dim = -1) * target_logits * self.empty_weight).sum(dim = -1)
                    loss_obj_ce = loss_obj_ce.sum() / self.empty_weight[target_classes].sum() 
                    # This is significant, we do not use .mean(). Instead, we aggregate weights for all gt labels.
                    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss 
                else:
                    loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
                losses = {'loss_obj_ce': loss_obj_ce}

                if log:
                    losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
                return losses

        elif self.obj_loss_type == 'cross_modal_matching':
            if self.subject_class:
                loss_sub_matching = self._contrastive_align(outputs, targets, indices, text_type = 'sub')
                loss_obj_matching = self._contrastive_align(outputs, targets, indices, text_type = 'obj')
                losses = {'loss_sub_matching': loss_sub_matching,\
                        'loss_obj_matching': loss_obj_matching}
                if log:
                    idx = self._get_src_permutation_idx(indices)
                    # idx: a tuple (batch_idx, src_idx)
                    target_classes_obj = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
                    target_classes_sub = torch.cat([t['sub_labels'][J] for t, (_, J) in zip(targets, indices)])
                    losses['obj_class_error'] = 100 - accuracy(outputs['pred_obj_logits'][idx], target_classes_obj)[0]
                    losses['sub_class_error'] = 100 - accuracy(outputs['pred_sub_logits'][idx], target_classes_sub)[0]
                return losses
            else:
                assert False
        else:
            assert False


    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits'] # [2, 100, 81]
        # print('pred_logits' + str(pred_logits.shape))
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        # tgt_lengths: number of predicted objects 
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # card_pred: number of true objects 
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # l1_loss that takes the mean element-wise absolute value difference.
        
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        if self.verb_loss_type in ['weighted_bce', 'focal', 'focal_bce', 'focal_without_sigmoid', 
                                   'asymmetric_bce', 'CB_focal_bce','bce']:
            assert 'pred_verb_logits' in outputs
            src_logits = outputs['pred_verb_logits'] # [2, 100, 117]

            idx = self._get_src_permutation_idx(indices)
            if self.giou_verb_label:
                _, cost_list = self.matcher(outputs, targets, return_cost = True) # cost_giou shape: [bs*num_queries, num_hoi]
                cost_giou = cost_list[0]
                cost_giou = - cost_giou # the matching giou used in QPIC is negative

                query_global = 0
                target_global = 0
                target_soft = []
                for t, (I, J) in zip(targets, indices):
                    ## relation label v1: only GIoU is considered.
                    soft_verb = ((cost_giou[query_global + I, target_global + J]) + 1) / 2 # scale giou to the range from 0 to 1

                    if ((soft_verb >= 0)&(soft_verb <= 1)).all().item() is False:
                        print(soft_verb)
                    assert ((soft_verb >= 0)&(soft_verb <= 1)).all()
                    if self.pseudo_verb:
                        assert 'target_verb_sim' in outputs.keys()
                        target_verb_sim = outputs['target_verb_sim']
                        # print((t['verb_labels'][J]>0).sum(), (target_verb_sim[target_global + J]>0).sum())
                        target_soft.append((t['verb_labels'][J] + target_verb_sim[target_global + J]) * soft_verb.unsqueeze(dim = -1))
                    else:
                        target_soft.append(t['verb_labels'][J] * soft_verb.unsqueeze(dim = -1))
                    query_global += src_logits.shape[1]
                    target_global += J.shape[0]
                target_classes_o = torch.cat(target_soft)
            elif self.naive_verb_smooth > 0:
                target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
                # target_classes_o = target_classes_o * (1 - self.naive_verb_smooth + self.naive_verb_smooth/src_logits.shape[-1])
                target_classes_o = target_classes_o * (1 - self.naive_verb_smooth + self.naive_verb_smooth/src_logits.shape[-1]) + \
                                (1 - target_classes_o) * self.naive_verb_smooth / src_logits.shape[-1]
            else:
                target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
            # [num_of_verbs, 117]
            if self.use_no_verb_token:
                src_logits = src_logits[:, :, :src_logits.shape[2]-1]
            target_classes = torch.zeros_like(src_logits)
            target_classes[idx] = target_classes_o

            if self.verb_loss_type == 'bce':
                loss_verb_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
            elif self.verb_loss_type == 'focal':
                src_logits = src_logits.sigmoid()
                if self.verb_curing:
                    src_logits = src_logits * outputs['curing_score']
                if self.giou_verb_label or self.naive_verb_smooth > 0:
                    loss_verb_ce = self._soft_neg_loss(src_logits, target_classes)
                else:
                    loss_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.verb_loss_type == 'focal_without_sigmoid':
                loss_verb_ce = self._neg_loss(src_logits, target_classes)
            elif self.verb_loss_type == 'focal_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._focal_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'asymmetric_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._asymmetric_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'CB_focal_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._CB_focal_bce(src_logits, target_classes)
            elif self.verb_loss_type == 'weighted_bce':
                src_logits = src_logits.sigmoid()
                loss_verb_ce = self._weighted_bce(src_logits, target_classes)

            if 'pri_pred_verb_logits' in outputs:
                pri_src_logits = outputs['pri_pred_verb_logits']
                if self.verb_loss_type == 'bce':
                    loss_verb_ce += F.binary_cross_entropy_with_logits(pri_src_logits, target_classes)
                elif self.verb_loss_type == 'focal':
                    pri_src_logits = pri_src_logits.sigmoid()
                    loss_verb_ce += self._neg_loss(pri_src_logits, target_classes)

            losses = {'loss_verb_ce': loss_verb_ce}
            return losses
        elif self.verb_loss_type in ['cross_modal_matching']:
            loss_verb_matching = self._contrastive_align(outputs, targets, indices, text_type = 'verb')
            losses = {'loss_verb_matching': loss_verb_matching}
            return losses
        else:
            assert False
    
    def loss_masked_entity_modeling(self, outputs, targets, indices, num_interactions):
        ### version 4, Recon modeling, proposed by Jianwen
        out_dict = {}
        loss_recon_labels = self.loss_obj_labels(outputs['recon_stat'], targets, indices, num_interactions)
        loss_recon_boxes = self.loss_sub_obj_boxes(outputs['recon_stat'], targets, indices, num_interactions)
        out_dict.update(loss_recon_labels)
        out_dict.update(loss_recon_boxes)
        new_out_dict = {i + '_recon': j for i,j in out_dict.items()}
        return new_out_dict
        
    
    def loss_gt_verb_recon(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        semantic = outputs['semantic']
        hs = outputs['hs']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o
        
        if self.verb_loss_type == 'bce':
            cls_loss = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        elif self.verb_loss_type == 'focal':
            src_logits = src_logits.sigmoid()
            # self.check_0_1(verb_gt_recon, 'src_logits')
            cls_loss = self._neg_loss(src_logits, target_classes)
        
        # Loss for All queries
        loss_recon = torch.tensor(0., device = target_classes.device)
        semantic_norm = norm_tensor(semantic)
        hs_norm = norm_tensor(hs)
        cos_sim = torch.einsum('abd,cd->abc', hs_norm, semantic_norm)
        pos_loss = 1 - cos_sim
        neg_loss = torch.clamp(cos_sim - 0.1, min = 0)
        recon_loss = (pos_loss * target_classes + neg_loss * (1 - target_classes)).sum() / target_classes.sum()

        loss = cls_loss + recon_loss

        return {'loss_verb_gt_recon': loss}


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        src_sub_boxes = outputs['pred_sub_boxes'][idx] # shape like [5, 4] [6, 4]...
        # print(src_sub_boxes)
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # print(target_sub_boxes)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        
        return losses
    
    def loss_kl_divergence(self, outputs, targets, indices, num_interactions):
        if 'verb_kl_divergence' in outputs:
            kl_param = outputs['verb_kl_divergence']
            bs, num_queries, latentdim2 = kl_param.shape # 2, 100, 256*2
            verb_mu, verb_log_var = kl_param[:,:,:latentdim2//2], kl_param[:,:,latentdim2//2:]
            verb_var = torch.exp(verb_log_var)
            loss = -0.5 * (1 + verb_log_var - verb_mu*verb_mu - verb_var)
            loss = torch.mean(loss)
            
        else:
            assert False

        return {'loss_kl_divergence': loss}

    def cal_entropy_loss(self, log_var, latentdim, bound):
        cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
        ### The following line seems to be a mistake in my previous implementation
        # var_term = 0.5*torch.sum(log_var, dim = 1) 
        var_term = 0.5*torch.sum(log_var, dim = -1)
        avg_entropy = torch.mean(cons_term + var_term) 
        loss = torch.max(torch.Tensor((0, bound - avg_entropy)).to(avg_entropy.device))
        return loss

    def loss_entropy_bound(self, outputs, targets, indices, num_interactions):
        if 'verb_log_var' in outputs:
            log_var = outputs['verb_log_var']
            b, nq, latentdim = log_var.shape
            latentdim = latentdim//2
            verb_log_var, obj_class_log_var = log_var[...,:latentdim], log_var[...,latentdim:]
            loss = self.cal_entropy_loss(verb_log_var, latentdim, bound = 256) +\
                   self.cal_entropy_loss(obj_class_log_var, latentdim, bound = 256)
            
        elif 'masked_context_log_var' in outputs:
            masked_memory_log_var = outputs['masked_context_log_var']
            _, latentdim = masked_memory_log_var.shape

            # Entropy bound
            cons_term = latentdim/2.*(math.log(2*math.pi) + 1.)
            var_term = 0.5*torch.sum(masked_memory_log_var, dim = 1) # [all pixels with false masks in all batches,]
            pixel_avg_entropy = torch.mean(cons_term + var_term)
            # print(pixel_avg_entropy)

            loss = torch.max(torch.Tensor((0, 256 - pixel_avg_entropy))).to(pixel_avg_entropy.device)

        else:
            assert False

        return {'loss_entropy_bound': loss}

        
    
    def loss_verb_hm(self, outputs, targets, indices, num_interactions):
        pred_verb_hm, mask = outputs['verb_hm']
        neg_loss = 0.
        # mask shape [bs,c,h,w]
        for ind, t in enumerate(targets):
            gt_verb_hm = t['verb_hm']
            valid_1 = torch.sum(~mask[ind][:,:,0])
            valid_2 = torch.sum(~mask[ind][:,0,:])
            # interpolate input [bs,c,h,w]
            gt_verb_hm = F.interpolate(gt_verb_hm.unsqueeze(0), size = (valid_1, valid_2)).squeeze(0)

            neg_loss += self._neg_loss(pred_verb_hm[ind][:,:valid_1,:valid_2], gt_verb_hm)

        return {'loss_verb_hm': neg_loss}
    
    def loss_verb_threshold(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        assert 'pred_verb_thr' in outputs
        src_logits = outputs['pred_verb_logits'] # [2, 100, 117]
        thr = outputs['pred_verb_thr'] # [2, 100, 117]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        # [num_of_verbs, 117]
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o  # [2, 100, 117]
        # target_classes = torch.ones_like(src_logits)*57
        # target_classes = torch.full(src_logits, self.num_verb_classes,
        #                             dtype=torch.int64, device=src_logits.device)

        sigma = torch.sigmoid(src_logits - thr)
        loss_verb_thr = self._neg_loss(sigma, target_classes, eps = 1e-6)

        return {'loss_verb_threshold': loss_verb_thr}


    def loss_semantic_similar(self, outputs, targets, indices, num_interactions):
        temperature = 0.05
        if 'semantic' in outputs and 'verb_verb_co' in outputs:
            # semantic = outputs['semantic_low'] # 117, 256
            semantic = outputs['semantic'] # 117, 256
            # verb_verb_co = outputs['verb_verb_co'] # 117, 117
            # verb_verb_co = outputs['joint_verb_verb_co'] # 117, 117
            # Symmetric cond prob
            verb_verb_co = outputs['verb_verb_co'] 
            verb_verb_co = verb_verb_co + verb_verb_co.T
            verb_verb_co = verb_verb_co / verb_verb_co.sum()

            norm_semantic = norm_tensor(semantic)
            # norm_semantic = semantic
            semantic_sim = torch.einsum('ab,cb->ac', norm_semantic, norm_semantic)  # 117, 117
            eye_mask = ~(torch.eye(verb_verb_co.shape[0], device = verb_verb_co.device) == 1)
            semantic_sim = semantic_sim[eye_mask]
            
            semantic_sim = F.log_softmax(semantic_sim / temperature)
            loss_sim = F.kl_div(semantic_sim, verb_verb_co[eye_mask], reduction = 'sum')

        else:
            loss_sim = torch.tensor([0.], device = outputs['pred_obj_logits'].device).sum()
        

        return {'loss_semantic_similar': loss_sim}
    
    def _weighted_bce(self, pred, gt, eps = 1e-6):

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = self.bce_weight[:,1]

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * pos_inds
        neg_loss = torch.log(1 - pred) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _CB_focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 2, beta = 0.9999):
        beta_weight = (1-beta) / (1 - torch.pow(beta, self.samples)) 
        beta_weight = beta_weight.unsqueeze(dim = 0).unsqueeze(dim = 0)

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds * beta_weight
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds * beta_weight

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _asymmetric_bce(self, pred, gt, eps = 1e-6, gamma_pos = 0, gamma_neg = 3, m = 0.01, vol = 1):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred_p = torch.clamp(pred, min = eps, max = 1.)
        pos_loss = torch.log(pred_p) * torch.pow(1 - pred_p, gamma_pos) * vol * pos_inds
        pred_m = torch.clamp(pred - m, min = 0, max = 1. - eps)
        neg_loss = torch.log(1 - pred_m) * torch.pow(pred_m, gamma_neg) * neg_weights * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss

    
    def _focal_bce(self, pred, gt, eps = 1e-6, gamma = 2, alpha = 0.5, vol = 4):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * alpha * vol * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * (1 - alpha) * vol * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss



    def _neg_loss(self, pred, gt, eps = 1e-6):
        ''' Modified focal loss. Exactly the same as CornerNet.
          Runs faster and costs a little bit more memory
        '''
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0
        pred = torch.clamp(pred, eps, 1.-eps)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum() 
        # It may appear to be nan, because there is -inf in torch.log(0)

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
    
    def _soft_neg_loss(self, pred, gt, eps = 1e-6, beta = 2):
        ''' Modified focal loss. Exactly the same as QFL.
        '''
        pos_inds = gt.gt(0).float()
        neg_inds = gt.eq(0).float()
        # print(pos_inds.sum(), neg_inds.sum())
        pred = torch.clamp(pred, eps, 1.-eps)
        loss = torch.pow(torch.abs(gt - pred), beta) * ((1 - gt) * torch.log(1 - pred) + gt * torch.log(pred))
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = - loss.sum()
        else:
            loss = - loss.sum() / num_pos


        return loss

    def _contrastive_align(self, outputs, targets, indices, text_type):
        '''
        indices: list of tensor tuples
        like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        '''
        assert self.use_no_verb_token == True  # The following implementation is based on this setting
        assert text_type in ['obj', 'sub', 'verb']
        logits_key = {'obj':'pred_obj_logits', 'sub':'pred_sub_logits', 'verb':'pred_verb_logits'}
        src_logits = outputs[logits_key[text_type]] / self.temperature

        if text_type in ['verb']:
            if sum([j.shape[0] for (_ , j) in indices]) > 0:
                idx = self._get_src_permutation_idx(indices)
                verb_labels = ()
                offset = 0
                global_idx = []
                max_text_len = max([v['verb_labels'].shape[1] for v in targets])
                max_len_tensor = None
                for t, (_, J) in zip(targets, indices):
                    # guard against squeeze_(dim = 0).unsqueeze_(dim = -1) operation
                    if t['verb_labels'].shape[0] > 0:
                        verb_labels += t['verb_labels'].split(1, dim = 0)
                        global_idx.append(J + offset)
                        offset += J.shape[0]
                    elif t['verb_labels'].shape[0] == 0 and t['verb_labels'].shape[1] == max_text_len:
                        # print('The sample with most verbs has zero triplets.')
                        max_len_tensor = torch.zeros((1, t['verb_labels'].shape[1]), device = t['verb_labels'].device)
                global_idx = torch.cat(global_idx)
                if max_len_tensor is not None:
                    verb_labels += (max_len_tensor,)
                for v in verb_labels:
                    v.squeeze_(dim = 0).unsqueeze_(dim = -1)
                tgt_verb_labels = pad_sequence(verb_labels).squeeze_(dim = -1).transpose(0, 1)
                if max_len_tensor is not None:
                    tgt_verb_labels = tgt_verb_labels[:tgt_verb_labels.shape[0]-1,:]
                # Pad the no_pred_token position of tgt_verb_labels to be 0, if self.no_pred_embedding is used in ParSeDETR
                zero_tensor = torch.zeros((tgt_verb_labels.shape[0], 1), device = tgt_verb_labels.device)
                tgt_verb_labels = torch.cat((tgt_verb_labels, zero_tensor), dim = 1)

                ############ The following setp is of GREAT importance ############
                ############ because we need to rearrange the order of the target labels before using positive_map[idx] = tgt_verb_labels.bool() ############
                tgt_verb_labels = tgt_verb_labels[global_idx]
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)
                # Replace the no_pred_token position of positive_map to be 1, if self.no_pred_embedding is used in ParSeDETR
                one_tensor = torch.ones((positive_map.shape[0], positive_map.shape[1]), device = positive_map.device)
                positive_map[:, :, positive_map.shape[2]-1] = one_tensor
                positive_map[idx] = tgt_verb_labels.bool()

            else:
                print('This batch (all samples) has zero triplets.')
                for t in targets:
                    if 'image_id' in t.keys():
                        print(f"image_id: {t['image_id']}")
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)

        elif text_type in ['obj', 'sub']:
            if sum([j.shape[0] for (_ , j) in indices]) > 0:
                idx = self._get_src_permutation_idx(indices)
                label_key = text_type + '_labels'  # 'obj_labels' or 'sub_labels'
                # idx: a tuple (batch_idx, src_idx)
                text_len = src_logits.shape[-1]
                target_classes_o = []
                for t, (_, J) in zip(targets, indices):
                    for j in J:
                        t_tensor = torch.zeros((text_len,))
                        t_tensor[t[label_key][j]] = 1
                        target_classes_o.append(t_tensor)
                        assert t_tensor.sum() == 1
                # Guard against no objects in all samples in this batch
                target_classes_o = torch.stack(target_classes_o, dim = 0).to(src_logits.device)
                # target_classes_o = torch.cat([t[label_key][J] for t, (_, J) in zip(targets, indices)])
                positive_map = torch.zeros(src_logits.shape, dtype = torch.bool).to(src_logits.device)
                # Replace the no_obj_token position of positive_map to be 1, if self.no_obj_embedding is used in ParSeDETR
                one_tensor = torch.ones((positive_map.shape[0], positive_map.shape[1]), device = positive_map.device)
                positive_map[:, :, positive_map.shape[2]-1] = one_tensor

                # 这个步骤应该不少了：这里缺少一步使用global idx对tgt_verb_labels顺序进行重新调整的过程，然后才能positive_map[idx] = tgt_verb_labels.bool()用这步骤
                # target_classes: init with a tensor of size src_logits.shape[:2]
                #                 and filled with self.num_obj_classes (no object class)
                positive_map[idx] = target_classes_o.bool()
            else:
                print('This batch (all samples) has zero ' + text_type + 's.')
                for t in targets:
                    if 'image_id' in t.keys():
                        print(f"image_id: {t['image_id']}")
                positive_map = torch.zeros(src_logits.shape, dtype=torch.bool).to(src_logits.device)
        
        positive_logits = -src_logits.masked_fill(~positive_map, 0)
        negative_logits = src_logits

        if self.matching_symmetric:
            # calculation of vis-to-text loss
            vis_with_pos = positive_map.any(dim = 2)
            pos_term = positive_logits.sum(dim = 2)
            neg_term = negative_logits.logsumexp(dim = 2)

            num_positive = positive_map.sum(dim = 2) + 1e-6

            vis_to_text_loss = (pos_term / num_positive + neg_term).masked_fill(~vis_with_pos, 0).sum()
            
            # calculation of text-to-vis loss
            text_with_pos = positive_map.any(dim = 1)
            pos_term = positive_logits.sum(dim = 1)
            neg_term = negative_logits.logsumexp(dim = 1)

            num_positive = positive_map.sum(dim = 1) + 1e-6

            text_to_vis_loss = (pos_term / num_positive + neg_term).masked_fill(~text_with_pos, 0).sum()
            
            return (vis_to_text_loss + text_to_vis_loss) / 2
        else:
            # print('None-symmetric')
            # calculation of vis-to-text loss
            vis_with_pos = positive_map.any(dim = 2)
            pos_term = positive_logits.sum(dim = 2)
            neg_term = negative_logits.logsumexp(dim = 2)

            num_positive = positive_map.sum(dim = 2) + 1e-6

            vis_to_text_loss = (pos_term / num_positive + neg_term).masked_fill(~vis_with_pos, 0).sum()
            
            return vis_to_text_loss


        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        
        # indices: list of tensor tuples
        # like [(tensor([ 5, 42, 51, 61]), tensor([2, 3, 0, 1])), (tensor([20]), tensor([0]))]
        
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            # 'verb_labels': self.loss_gt_verb_recon,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'entropy_bound':self.loss_entropy_bound,
            'kl_divergence':self.loss_kl_divergence,
            'verb_hm':self.loss_verb_hm,
            'semantic_similar':self.loss_semantic_similar,
            'verb_threshold':self.loss_verb_threshold,
            'masked_entity_modeling':self.loss_masked_entity_modeling,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        if self.triplet_filtering:
            bs, num_query = outputs['pred_verb_logits'].shape[:2]
            cost_triplet = {bs_i:{} for bs_i in range(bs)}
            indices, cost_list = self.matcher(outputs_without_aux, targets, return_cost = True)
            C = 1 * cost_list[6] + 1 * cost_list[5] + 1 * cost_list[4] + \
                2.5 * cost_list[2] + 1 * cost_list[0]
            
            query_global = 0
            target_global = 0
            for bs_i, (I, J) in enumerate(indices):
                # target_cost = C[query_global + I, target_global + J] # scale giou to the range from 0 to 1
                for I_i, J_i in zip(I, J):
                    assert J_i not in cost_triplet[bs_i].keys()
                    cost_triplet[bs_i][int(J_i)] = C[query_global + I_i, target_global + J_i]
                query_global += num_query
                target_global += J.shape[0]
            


            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices, cost_list_i = self.matcher(aux_outputs, targets, return_cost = True)
                C_i = 1 * cost_list_i[6] + 1 * cost_list_i[5] + 1 * cost_list_i[4] + \
                      2.5 * cost_list_i[2] + 1 * cost_list_i[0]

                query_global = 0
                target_global = 0
                for bs_i, (I, J) in enumerate(indices):
                    # target_cost = C_i[query_global + I, target_global + J] # scale giou to the range from 0 to 1
                    for I_i, J_i in zip(I, J):
                        cost_triplet[bs_i][int(J_i)] += C_i[query_global + I_i, target_global + J_i]
                    query_global += num_query
                    target_global += J.shape[0]
            
            ## Perform outlier detection
            cost_triplet_list = []
            for c in cost_triplet.values():
                cost_triplet_list += list(c.values())
            if len(cost_triplet_list) > 0:
                cost_triplet_list = torch.stack(cost_triplet_list)
                up_thre = torch.mean(cost_triplet_list) + torch.std(cost_triplet_list) * 0.5
                
                flag_dict = {} # We keep it if it's True
                # gt_sum = 0
                # keep_sum = 0
                for bs_i, c in cost_triplet.items():
                    flag_i = torch.ones((len(c),), device = outputs['pred_verb_logits'].device).bool()
                    # gt_sum += len(c)
                    for j, c_j in c.items():
                        flag_i[j] = (c_j <= up_thre)
                    flag_dict[bs_i] = flag_i
                    # keep_sum += flag_i.float().sum()
                # print('Keeping ratio: {:.2f}'.format(keep_sum/gt_sum))

                for bs_i, t in enumerate(targets):
                    t['obj_labels'] = t['obj_labels'][flag_dict[bs_i]]
                    t['sub_labels'] = t['sub_labels'][flag_dict[bs_i]]
                    t['verb_labels'] = t['verb_labels'][flag_dict[bs_i]]
                    t['sub_boxes'] = t['sub_boxes'][flag_dict[bs_i]]
                    t['obj_boxes'] = t['obj_boxes'][flag_dict[bs_i]] 
            
            

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'verb_hm':
                        continue
                    kwargs = {}
                    if loss == 'obj_labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses