import torch
from torch.nn import CrossEntropyLoss
from torch.nn import L1Loss

from Code.Utilities.loading_utils import load


def similarity_loss(response_map):

    """ Loss function enforcing feature similarity """

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    # Compute cross entropy
    loss = CrossEntropyLoss()(response_map, index)

    return loss


def continuity_loss(response_map, shape_x, shape_y):

    """ Loss function enforcing spatial continuity"""

    # Load parameters
    parameters = load("architecture.yaml")
    globals().update(parameters)

    # Initiate loss functions for vertical/horizontal continuity
    loss_vertical   = L1Loss(size_average=True)
    loss_horizontal = L1Loss(size_average=True)

    # Reshape response map back into its original shape
    response = response_map.reshape((shape_y, shape_x, n_classes))

    # Difference of (shifted) response maps (vectorial implementation of formula on page 5)
    vertical_subtraction   = response[1:, :] - response[:-1, :]
    horizontal_subtraction = response[:, 1:] - response[:, :-1]

    # Compute target for loss functions (ideally they should be zero)
    zero_target_vertical   = torch.zeros(shape_y - 1, shape_x, n_classes)
    zero_target_horizontal = torch.zeros(shape_y, shape_x - 1, n_classes)

    if torch.cuda.is_available():

        # Pass to GPU
        zero_target_vertical   = zero_target_vertical.cuda()
        zero_target_horizontal = zero_target_horizontal.cuda()

    # Compute loss functions along the two axis
    vertical_loss   = loss_vertical(vertical_subtraction, zero_target_vertical)
    horizontal_loss = loss_horizontal(horizontal_subtraction, zero_target_horizontal)

    return vertical_loss + horizontal_loss


def loss_function(response_map, shape_x, shape_y):

    """ Composite loss function with mu, nu weight for the two component loss functions"""

    # Load weighting parameters
    parameters = load("loss.yaml")
    globals().update(parameters)

    # Compute loss function
    loss = mu*similarity_loss(response_map) + nu*continuity_loss(response_map, shape_x, shape_y)

    return loss
