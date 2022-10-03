import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Image():

    # Constructor
    def __init__(self, path):

        """ Initialise with path only"""

        # Store image path
        self.path = path

        # Read image
        self.read()

        # Store properties
        self.store_properties()

        # Read normalised image
        self.read_normalised_image()

        # Convert to 8-bit
        self.to_8bit()

        # Convert to data
        self.to_data()


    def read(self):

        """ Read image """

        # Read image
        self.image = cv2.imread(self.path)


    def store_properties(self):

        """ Store relevant properties of image at hand """

        # Store max and min value
        self.max = (self.image).max()
        self.min = (self.image).min()

        # Store span
        self.span = self.max - self.min

        # Store shape
        self.shape = (self.image).shape

        # Store type
        self.type = (self.image).dtype

        # Ensure presence of CUDA
        self.cuda = torch.cuda.is_available()


    def read_normalised_image(self):

        """ Read and store the normalised image """

        # Convert image to float
        image = (self.image).astype("float32")

        # "Centre" image
        centred = image - self.min

        # Rescale image
        scaled = centred / (self.span * 1.0)

        # Store rescaled image
        self.normalised_image = scaled



    def to_8bit(self):

        """ Convert 16-bit image to 8-bit """

        # Convert to 8-bit
        image_8b = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Convert to 8-bit
        #image_8b_norm = cv2.normalize(self.normalised_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        image_8b_norm = self.normalised_image
        # Store the 8-bit image
        self.image_8b = image_8b

        # Store the 8-bit image
        self.normalised_image_8b = image_8b_norm.astype('float32')


    def to_data(self):

        """ Convert to data usable by the Neural Network """

        # Import image from numpy
        data = torch.from_numpy(np.array([self.normalised_image_8b.transpose((2, 0, 1))]))

        # Activate CUDA if possible
        if self.cuda:
            data = data.cuda()

        self.data = Variable(data)


    def plot(self):

        """ Plot image """

        figsize = (10, 10)

        plt.figure(figsize=figsize)
        plt.imshow(self.image)
        plt.axis("off")
        plt.show()


