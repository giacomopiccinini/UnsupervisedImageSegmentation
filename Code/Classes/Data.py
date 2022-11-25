from glob import glob
from pathlib import Path
from torch.utils.data import Dataset

from Code.Classes.Image import Image


class Data(Dataset):

    """Define a custom Data class based on torch Dataset."""

    # These three methods always need to be specified

    def __init__(self, path):

        """Constructor"""

        # Store admissible image formats
        self.formats = [".png", ".jpg", ".jpeg", ".tif"]

        # Read path from command line
        path = Path(path).absolute()

        # Check if argument is a file or a directory (and check if it exists)
        # If path is a directory, load all files recursively
        if path.is_dir():
            files = list(path.rglob("*"))
        # If is a single file, load it alone
        elif path.is_file():
            files = [path]
        else:
            raise Exception(f"ERROR: {path} does not exist")

        # Ensure only compliant files are considered
        files = [file for file in files if file.suffix in self.formats]

        # Store info
        self.path = path
        self.n_images = len(files)
        self.names = [file.name for file in files]
        self.images = [file.__str__() for file in files]

    def __len__(self):

        """Return number of images in the dataset"""

        return self.n_images

    def __getitem__(self, index):

        """Specify how to retrieve images"""

        # Retrieve image path
        image_path = self.images[index]

        # Create Image class instance
        image = Image(image_path)

        # Return the actual image (i.e. numpy array) and its name
        return image.data[0], self.names[index]
