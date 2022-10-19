import torch
from torch.nn import CrossEntropyLoss
from torch.nn import L1Loss


def similarity_loss(response_map):

    """Loss function enforcing feature similarity"""

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    # Compute cross entropy
    loss = CrossEntropyLoss()(response_map, index)

    return loss


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


def max_area_loss(response_map):

    """Loss function enforcing feature similarity"""

    # Extrapolate info on images
    batch_size, n_classes, shape_y, shape_x = response_map.shape

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    # Compute areas of distinct regions
    areas = torch.bincount(torch.flatten(index))

    # Compute the total area of the array
    normalization_factor = n_classes * shape_x * shape_y

    # Find maximal area
    max_area = torch.max(areas)

    # Return the normalized maximum
    return max_area / normalization_factor


def min_area_loss(response_map):

    """Loss function enforcing feature similarity"""

    # Extrapolate info on images
    batch_size, n_classes, shape_y, shape_x = response_map.shape

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    # Compute areas of distinct regions
    areas = torch.bincount(torch.flatten(index))

    # Restrict to non-zero areas
    areas = areas[areas.nonzero()]

    # Compute the total area of the array
    normalization_factor = n_classes * shape_x * shape_y

    # Find maximal area
    min_area = torch.min(areas)

    return 1 - (min_area / normalization_factor)


def loss_function(response_map, args):

    """Composite loss function with mu, nu weight for the two component loss functions"""

    # Compute loss function
    loss = (
        args.mu * similarity_loss(response_map)
        + args.nu * continuity_loss(response_map, args)
        + 0.5 * max_area_loss(response_map)
        + min_area_loss(response_map)
    )

    return loss
