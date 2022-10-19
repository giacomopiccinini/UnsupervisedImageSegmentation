import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):

    # Construct for SegNet class
    def __init__(self, args):

        # Initialise using nn class
        super(SegNet, self).__init__()

        self.input_dim      = args.n_channels
        self.n_features     = args.n_features
        self.n_classes      = args.n_classes
        self.n_convolutions = args.n_convolutions

        # Add FIXED top layers to architecture:

        # Add convolutional layer
        self.conv_fixed_top = nn.Conv2d(self.input_dim, self.n_features, kernel_size=3, stride=1, padding=1)

        # Add batch normalisation layer
        self.bn_fixed_top   = nn.BatchNorm2d(self.n_features)

        # Declare other convolutional/batchnorm layer to be (dynamic) lists depending on input
        self.conv_dynamic = nn.ModuleList()
        self.bn_dynamic   = nn.ModuleList()

        for i in range(self.n_convolutions - 1):

            # Add dynamic convolution layer
            self.conv_dynamic.append(nn.Conv2d(self.n_features, self.n_features, kernel_size=3, stride=1, padding=1))

            # Add dynamic batch normalisation layer
            self.bn_dynamic.append(nn.BatchNorm2d(self.n_features))

        # Add FIXED bottom layers to architecture:
        # Add convolutional layer to reduce to n_classes
        self.conv_fixed_bottom = nn.Conv2d(self.n_features, self.n_classes, kernel_size=1, stride=1, padding=0)

        # Add batch normalisation layer
        self.bn_fixed_bottom = nn.BatchNorm2d(self.n_classes)


    def forward(self, x):

        """ Define forward passing operation """

        # Apply first convolution
        x = self.conv_fixed_top(x)

        # Apply relu activation function
        x = F.relu(x)

        # Apply batch normalisation
        x = self.bn_fixed_top(x)

        # Loop over intermediate dynamic layers
        for i in range(self.n_convolutions - 1):

            # Apply convolutional layer
            x = self.conv_dynamic[i](x)

            # Apply activation function
            x = F.relu(x)

            # Apply batch normalisation
            x = self.bn_dynamic[i](x)

        # Apply last convolution
        x = self.conv_fixed_bottom(x)

        # Apply last batch normalisation
        x = self.bn_fixed_bottom(x)

        return x