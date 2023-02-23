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
        drop_last (bool): Whether to drop last batch
    '''
    def __init__(self, labels, n_classes, n_instances, drop_last = False):
        
        self.classes = labels.unique().numpy()
        self.labels = labels
        self.n_classes = n_classes     
        self.n_instances = n_instances
        self.drop_last = drop_last

        self.iters = len(labels) // (n_classes * n_instances)
        self.batch_size = n_classes * n_instances
        
    def __len__(self):
        return self.iters if self.drop_last else self.iters + 1

    def __iter__(self):

        classes = np.random.permutation(self.classes)
        label_indices = torch.arange(len(self.labels))
        used_indices = []

        for index in range(self.iters):
            indices = []
            selected_classes = np.random.choice(classes, size = self.n_classes, replace = False)
            for c in selected_classes:
                random_indices = np.random.permutation((self.labels == c).nonzero().flatten())
                random_indices = np.setdiff1d(random_indices, used_indices)
                selected_indices = random_indices[:self.n_instances]
                indices.extend(selected_indices)
                used_indices.extend(indices)
            if len(indices) < self.batch_size:
                missing = self.batch_size - len(indices)
                missing_indices = np.random.choice(np.setdiff1d(label_indices, used_indices), size = missing, replace = False)
                indices.extend(missing_indices)
                used_indices.extend(indices)
            label_indices = np.setdiff1d(label_indices, used_indices)

            yield indices
        
        if len(label_indices) > 0 and not self.drop_last: # Last batch
            yield label_indices