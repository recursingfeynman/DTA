import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler

class MPerClassSampler(BatchSampler):
    '''
    Prepares indices. Each batch will contain N classes of M samples
    Args:
        labels (torch.tensor): True labels
        n_classes (int): How many classes use for batch construction
        n_instances (int): How many samples per class batch should contain
    '''
    def __init__(self, labels, n_classes, n_instances):
        
        self.classes = labels.unique().numpy()
        self.labels = labels
        self.n_classes = n_classes     
        self.n_instances = n_instances
        
    def __len__(self):
        return len(self.labels) // (self.n_classes * self.n_instances)

    def __iter__(self):

        n_classes = self.n_classes
        n_instances = self.n_instances
        labels = self.labels
        classes = np.random.permutation(self.classes)
        label_indices = torch.arange(len(self.labels))

        used_indices = []

        batch_size = n_classes * n_instances
        iters = len(labels) // (n_classes * n_instances)
        for index in range(iters):
            
            indices = []

            selected_classes = np.random.choice(classes, size = self.n_classes, replace = False)

            for c in selected_classes:
                random_indices = np.random.permutation((labels == c).nonzero().flatten())
                random_indices = np.setdiff1d(random_indices, used_indices)
                selected_indices = random_indices[:n_instances]
                indices.extend(selected_indices)
                used_indices.extend(indices)
            
            if len(indices) < batch_size:
                missing = batch_size - len(indices)
                missing_indices = np.random.choice(np.setdiff1d(label_indices, used_indices), size = missing, replace = False)
                indices.extend(missing_indices)
                used_indices.extend(indices)

            label_indices = np.setdiff1d(label_indices, used_indices)

            yield indices

        if len(label_indices) > 0:
            yield label_indices