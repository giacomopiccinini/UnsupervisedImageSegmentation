import torch


def nn(mask_1: torch.tensor, mask_2: torch.tensor, k: int) -> torch.tensor:

    """For each point in mask_1 the k nearest neighbour belonging to mask_2 are found.
    If mask_1 comprises of n_1 non-trivial (i.e. non-zero) points, the output is a numpy array of shape (n_1, k, 2).
    That is, for each non-trivial point in mask_1, we find the nearest k points and return their coordinates.
    The metric used is Manhattan."""

    # Extract coordinates of points of mask_1
    mask_1_coordinates = torch.argwhere(mask_1)

    # Extract coordinates of points of mask_2
    mask_2_coordinates = torch.argwhere(mask_2)

    # Compute differences along x and y
    difference = mask_1_coordinates[:, None] - mask_2_coordinates

    # Compute the moduli of difference along each direction
    moduli = torch.abs(difference)

    # Sum moduli to obtain the actual Manhattan metric
    metrics = torch.sum(moduli, axis=2)

    # Sort points by distance
    ordered_distances = torch.argsort(metrics, axis=1)

    # Keep only the first k neighbours and obtain indices of their coordinates
    nn_indices = ordered_distances[:, :k]

    # Retrieve actual coordinates
    nn_coordinates = mask_2_coordinates[nn_indices]

    return nn_coordinates


def nn_mask(mask_1: torch.tensor, mask_2: torch.tensor, k: int) -> torch.tensor:

    """For each point in mask_1 the k nearest neighbour belonging to mask_2 are found.
    The output is a numpy array of the same shape as mask_1 and mask_2 but with only nearest neighbours (in mask_2)
    to each and every point of mask_1 are set to 1.
    The metric used is Manhattan."""

    # Retrieve coordinates
    nn_coordinates = nn(mask_1, mask_2, k)

    # Reshape coordinates list
    nn_coordinates = nn_coordinates.reshape(-1, 2)

    # Create blank mask
    mask = torch.zeros_like(mask_1)

    # Fill in blank mask
    mask[tuple(zip(*nn_coordinates))] = 1

    return mask


def nn_value(
    image: torch.tensor, mask_1: torch.tensor, mask_2: torch.tensor, k: int
) -> torch.tensor:

    """Retrieve alues of k nearest neighbours"""

    # Find NN coordinates
    nn_coordinates = nn(mask_1, mask_2, k)

    # Retrieve NN values (syntax is convoluted)
    nn_values = image[tuple(nn_coordinates.T.tolist())].T

    return nn_values
