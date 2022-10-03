import torch
import numpy as np
import cv2
from Code.Utilities.loading_utils import load

def plot(image, response_map):

    """ Plot segmented image as the model is trained """

    # Load training parameters
    architecture_parameters = load("architecture.yaml")
    globals().update(architecture_parameters)

    # Apply argmax on every row, keep only index (= class), throw away value
    _, index = torch.max(response_map, 1)

    im_target = index.data.cpu().numpy()
    labels = len(np.unique(im_target))
    n_classes = len(np.unique(im_target))

    label_colours = np.random.randint(255, size=(n_classes, 3))

    im_target_rgb = np.array([label_colours[c % n_classes] for c in im_target])
    im_target_rgb = im_target_rgb.reshape(image.shape).astype(np.uint8)
    cv2.imwrite("Output/segmentation.png", im_target_rgb)
    cv2.imshow("output", im_target_rgb)
    cv2.waitKey(10)

    return labels