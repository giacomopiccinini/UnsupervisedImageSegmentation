from Code.Modules.A_train import train
from Code.Classes.Image import Image

import sys

# Check arguments
if len(sys.argv) != 2:
    print("Please use 'python main.py <FILENAME>'")
    exit(1)

# Read filename from command line
filename = sys.argv[1]

# Initialise image
image = Image(f"Input/{filename}")

train(image)
