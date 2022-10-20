import torch
import cv2
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from Code.Classes.Data import Data

def plot(model, args):

    """ Plot final segmented model"""    

    use_cuda = torch.cuda.is_available()

    # Enable CUDA
    if use_cuda:
        model.cuda()
    model.train()
        
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    model.to(device = device)

    # Set model to evaluation to make sure layers behave correctly in inference
    model.eval()

    # Load dataset and loader for PyTorch
    dataset = Data(args.path)
    loader = DataLoader(dataset)

    for batch, names in tqdm(loader):
        
        batch = batch.to(device = device)

        # Pass image through SegNet
        response_map = model(batch)

        # Apply argmax on every row, keep only index (= class), throw away value
        _, indexed_images = torch.max(response_map, 1)

        # Convert the batch of images into numpy arrays
        segmented_images = indexed_images.data.cpu().numpy()

        # Ensure data type is correct (they are just masks)
        segmented_images = segmented_images.astype('uint8')

        for segmented_image, name in zip(segmented_images, names):

            # Find number of distinct classes
            n_classes = len(np.unique(segmented_image))

            # Assign them a random colour
            label_colours = np.random.randint(255, size=(n_classes, 3))

            # Colour the image accordingly
            segmented_image = np.array([label_colours[c % n_classes] for c in segmented_image])

            # Save images
            cv2.imwrite(f"Output/segmentation.png", segmented_image)
