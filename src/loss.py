# This section was inspired by tensorflow TripletLoss implementation:
# https://www.tensorflow.org/addons/api_docs/python/tfa/losses/TripletSemiHardLoss

import torch
from utils import pairwise_cosine

def masked_minimum_torch(data, mask, dim = 1):
    axis_maximus, _ = torch.max(data, dim, keepdim = True)
    
    masked_minimus = (
        torch.min(
        torch.multiply(data - axis_maximus, mask), dim, keepdim = True
        )[0] 
        + axis_maximus
    )

    return masked_minimus

def masked_maximum_torch(data, mask, dim = 1):
    axis_minimus, _ = torch.min(data, dim, keepdim = True)
    masked_maximus = (
        torch.max(
            torch.multiply(data - axis_minimus, mask), dim, keepdim = True
        )[0]
        + axis_minimus
        )

    return masked_maximus

class TripletLossHard(torch.nn.Module):
    def __init__(self, margin = 1.0, soft = None):
        super().__init__()
        self.margin = margin
        self.soft = soft
        
    def forward(self, embeddings, labels):
        
        if embeddings.is_cuda:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        batch_size = labels.size(0)
        
        dist = 1 - pairwise_cosine(embeddings, zero_diag = False)
        
        mask_positive = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        mask_negative = torch.logical_not(mask_positive).float()

        hard_negative = masked_minimum_torch(dist, mask_negative)
    
        mask_positive = mask_positive - torch.eye(batch_size, device = device)
        hard_positive = masked_maximum_torch(dist, mask_positive)
    
        if self.soft:
            triplet_loss = torch.log1p(torch.exp(hard_positive - hard_negative))
        else:
            triplet_loss = torch.nn.functional.relu(hard_positive - hard_negative + self.margin)
        
        triplet_loss = torch.mean(triplet_loss)
        
        return triplet_loss
