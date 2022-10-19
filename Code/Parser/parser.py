def parse():

    """ Parse command line arguments """

    import argparse

    # Initiate argparser
    parser = argparse.ArgumentParser()

    # Parse path
    parser.add_argument('--path', const="Input", default = "Input", nargs='?', type=str, help='String for path to input file or directory')
    parser.add_argument('--n_channels', const=3, default = 3, nargs='?', type=int, help='Number of channels (colours) of input images')

    # Parse architecture parameters
    parser.add_argument('--n_features', const=100, default = 100, nargs='?', type=int, help='Integer for number of features in latent space')
    parser.add_argument('--n_classes', const=15, default = 15, nargs='?', type=int, help='Integer for number of classes we begin with')
    parser.add_argument('--min_classes', const=3, default = 3, nargs='?', type=int, help='Integer for minimum number of classes to end up with')
    parser.add_argument('--n_convolutions', const=2, default = 2, nargs='?', type=int, help='Integer for number of convolutions to apply')
    parser.add_argument('--batch_size', const=16, default = 16, nargs='?', type=int, help='Integer for batch size')

    # Parse training parameters
    parser.add_argument('--learning_rate', const=0.1, default = 0.1, nargs='?', type=float, help='Float for learning rate during training')
    parser.add_argument('--momentum', const=0.9, default = 0.9, nargs='?', type=float, help='Float for momentum during training')
    parser.add_argument('--epochs', const=30, default = 30, nargs='?', type=int, help='Integer for  number of epochs during training')

    # Parse loss function parameters
    parser.add_argument('--mu', const=2, default = 2, nargs='?', type=int, help='Integer for weight of similarity loss function')
    parser.add_argument('--nu', const=2, default = 2, nargs='?', type=int, help='Integer for weight of continuity loss function (overall)')
    parser.add_argument('--nu_x', const=1, default = 1, nargs='?', type=int, help='Integer for weight of continuity loss function along x')
    parser.add_argument('--nu_y', const=1, default = 1, nargs='?', type=int, help='Integer for weight of continuity loss function along y')

    # Parse model parameters
    parser.add_argument('--model_path', const="Model/model.pt", default = "Model/model.pt", nargs='?', type=str, help='String for saved model')

    args = parser.parse_args()

    return args