from Code.Loss.continuity import continuity_loss
from Code.Loss.similarity import similarity_loss
from Code.Loss.max_area import max_area_loss
from Code.Loss.min_area import min_area_loss


def loss_function(response_map, args):

    """Composite loss function with mu, nu weight for the two component loss functions"""

    # Compute loss function
    loss = (
        args.mu * similarity_loss(response_map)
        + args.nu * continuity_loss(response_map, args)
        + args.M * max_area_loss(response_map)
        + args.m * min_area_loss(response_map)
    )

    return loss
