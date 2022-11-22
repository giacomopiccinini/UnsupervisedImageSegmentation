import torch
from torch.nn import L1Loss


def continuity_loss(response_map, args):

    """Loss function enforcing spatial continuity"""

    # Extrapolate info on images
    batch_size, n_classes, shape_y, shape_x = response_map.shape

    # Initiate loss functions for vertical/horizontal continuity
    loss_vertical = L1Loss(reduction="mean")
    loss_horizontal = L1Loss(reduction="mean")

    # Difference of (shifted) response maps (vectorial implementation of formula on page 5)
    vertical_subtraction = response_map[:, :, 1:, :] - response_map[:, :, :-1, :]
    horizontal_subtraction = response_map[:, :, :, 1:] - response_map[:, :, :, :-1]

    # Compute target for loss functions (ideally they should be zero)
    zero_target_vertical = torch.zeros(batch_size, n_classes, shape_y - 1, shape_x)
    zero_target_horizontal = torch.zeros(batch_size, n_classes, shape_y, shape_x - 1)

    if torch.cuda.is_available():

        # Pass to GPU
        zero_target_vertical = zero_target_vertical.cuda()
        zero_target_horizontal = zero_target_horizontal.cuda()

    # Compute loss functions along the two axis
    vertical_loss = loss_vertical(vertical_subtraction, zero_target_vertical)
    horizontal_loss = loss_horizontal(horizontal_subtraction, zero_target_horizontal)

    return args.nu_y * vertical_loss + args.nu_x * horizontal_loss
