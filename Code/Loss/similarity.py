import torch
from torch.nn import CrossEntropyLoss


def similarity_loss(response_map):

    """Loss function enforcing feature similarity"""

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    # Compute cross entropy
    loss = CrossEntropyLoss()(response_map, index)

    return loss
