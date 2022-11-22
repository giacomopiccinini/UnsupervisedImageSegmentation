import torch


def min_area_loss(response_map):

    """Loss function enforcing feature similarity"""

    # Extrapolate info on images
    _, n_classes, shape_y, shape_x = response_map.shape

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
