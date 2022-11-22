import torch


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
