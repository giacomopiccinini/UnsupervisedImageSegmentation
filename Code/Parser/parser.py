def parse():

    """Parse command line arguments"""

    import argparse

    # Initiate argparser
    parser = argparse.ArgumentParser()

    # Parse path
    parser.add_argument(
        "--path",
        const="Input",
        default="Input",
        nargs="?",
        type=str,
        help="String for path to input file or directory",
    )
    parser.add_argument(
        "--n_channels",
        const=3,
        default=3,
        nargs="?",
        type=int,
        help="Number of channels (colours) of input images",
    )

    # Parse architecture parameters
    parser.add_argument(
        "--n_features",
        const=100,
        default=100,
        nargs="?",
        type=int,
        help="Integer for number of features in latent space",
    )
    parser.add_argument(
        "--n_classes",
        const=100,
        default=100,
        nargs="?",
        type=int,
        help="Integer for number of classes we begin with",
    )
    parser.add_argument(
        "--min_classes",
        const=3,
        default=3,
        nargs="?",
        type=int,
        help="Integer for minimum number of classes to end up with",
    )
    parser.add_argument(
        "--n_convolutions",
        const=2,
        default=2,
        nargs="?",
        type=int,
        help="Integer for number of convolutions to apply",
    )
    parser.add_argument(
        "--batch_size",
        const=16,
        default=16,
        nargs="?",
        type=int,
        help="Integer for batch size",
    )

    # Parse training parameters
    parser.add_argument(
        "--learning_rate",
        const=0.1,
        default=0.1,
        nargs="?",
        type=float,
        help="Float for learning rate during training",
    )
    parser.add_argument(
        "--momentum",
        const=0.4,
        default=0.4,
        nargs="?",
        type=float,
        help="Float for momentum during training",
    )
    parser.add_argument(
        "--epochs",
        const=50,
        default=50,
        nargs="?",
        type=int,
        help="Integer for  number of epochs during training",
    )

    # Parse loss function parameters
    parser.add_argument(
        "--mu",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Float for weight of similarity loss function",
    )
    parser.add_argument(
        "--nu",
        const=2,
        default=2,
        nargs="?",
        type=float,
        help="Float for weight of continuity loss function (overall)",
    )
    parser.add_argument(
        "--nu_x",
        const=1,
        default=1,
        nargs="?",
        type=float,
        help="Float for weight of continuity loss function along x",
    )
    parser.add_argument(
        "--nu_y",
        const=1,
        default=1,
        nargs="?",
        type=float,
        help="Float for weight of continuity loss function along y",
    )

    parser.add_argument(
        "--M",
        const=1,
        default=1,
        nargs="?",
        type=float,
        help="Float for weight of max area loss function",
    )

    parser.add_argument(
        "--m",
        const=1,
        default=1,
        nargs="?",
        type=float,
        help="Float for weight of min area loss function",
    )

    # Parse model parameters
    parser.add_argument(
        "--model_path",
        const="Model/model.pt",
        default="Model/model.pt",
        nargs="?",
        type=str,
        help="String for saved model",
    )

    # Show results of segmentation in real time
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show (or not) the results of segmentation",
    )

    # Show pie chart
    parser.add_argument(
        "--pie",
        action="store_true",
        help="Show (or not) the pie chart of segmentation",
    )

    # Show pie chart
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix (or not) the boundary",
    )

    # Force reduction to n classes
    parser.add_argument(
        "--force",
        const=2,
        default=2,
        nargs="?",
        type=int,
        help="Integer indicating whether to force reduction to n classes (based on NNs)",
    )

    args = parser.parse_args()

    return args
