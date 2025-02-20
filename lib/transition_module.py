import numpy as np
import torch
import torch.nn as nn


class transition_module(nn.Module):
    # 区分obj
    def __init__(self):

        super(transition_module, self).__init__()

        a_trans_matrix = torch.rand((37, 3, 3), requires_grad=True)
        s_trans_matrix = torch.rand((37, 6, 6), requires_grad=True)
        c_trans_matrix = torch.rand((37, 17, 17), requires_grad=True)

        a_trans_matrix = nn.functional.softmax(a_trans_matrix, dim=2)
        s_trans_matrix = nn.functional.softmax(s_trans_matrix, dim=2)
        c_trans_matrix = nn.functional.softmax(c_trans_matrix, dim=2)

        self.a_trans_matrix = torch.nn.Parameter(a_trans_matrix)
        self.s_trans_matrix = torch.nn.Parameter(s_trans_matrix)
        self.c_trans_matrix = torch.nn.Parameter(c_trans_matrix)

    def forward(self, spatial_label, contact_label, obj_label):

        pred_spatial_label = torch.matmul(spatial_label, nn.functional.softmax(self.s_trans_matrix, dim=2))[obj_label, list(range(len(obj_label)))]
        pred_contact_label = torch.matmul(contact_label, nn.functional.softmax(self.c_trans_matrix, dim=2))[obj_label, list(range(len(obj_label)))]

        return pred_spatial_label, pred_contact_label


    def __convert(self, freq):
        # convert 37*rel_num to 37*rel_num*rel_num
        obj_num, rel_num = freq.shape
        freq_matrix = np.zeros([obj_num, rel_num, rel_num])
        for i in range(obj_num):
            if i >= 2:
                for j in range(rel_num):
                    freq_matrix[i][j] = freq[i]
        return freq_matrix
