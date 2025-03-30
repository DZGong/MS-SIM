import tifffile
import numpy as np
import matplotlib.pyplot as plt

class SIMDataset:
    """
    A class used to represent a Structured Illumination Microscopy (SIM) dataset.

    Attributes
    ----------
    original_data : ndarray
        a 4D array containing the original image data
    num_channels : int
        the number of channels in the image data
    num_slices : int
        the number of slices in the image data
    full_image_height : int
        the height of the full images
    image_width : int
        the width of the images
    num_subimages : int
        the number of subimages to split each image into
    subimage_height : int
        the height of each subimage
    data : ndarray
        a 5D array containing the subimage data

    Methods
    -------
    __repr__():
        Returns a string representation of the SIMDataset instance.
    __len__():
        Returns the total number of subimages in the dataset.
    __getitem__(idx):
        Returns a specific subimage from the dataset.
    create_subimages():
        Splits each image into subimages and stores them in the data array.
    """
    def __init__(self, filename):
        """Initialize the dataset with the filename of the TIFF file."""
        self.original_data = tifffile.imread(filename)
        self.num_channels = self.original_data.shape[0]
        self.num_slices = self.original_data.shape[1]
        self.full_image_height = self.original_data.shape[2]
        self.image_width = self.original_data.shape[3]
        self.num_subimages = 3
        self.subimage_height = self.full_image_height // self.num_subimages

        # Initialize an empty array for the subimages
        self.data = np.empty((
            self.num_channels,
            self.num_slices,
            self.num_subimages,
            self.subimage_height,
            self.image_width
            ),
            dtype=self.original_data.dtype
        )

        # Fill the subimages array
        self.create_subimages()

    def __repr__(self):
        return (
            f"SIMDataset(channels={self.num_channels}, "
            f"slices={self.num_slices}, subimages={self.num_subimages}, "
            f"image_dimensions=({self.subimage_height}, {self.image_width}))"
        )

    def __len__(self):
        return self.num_channels * self.num_slices * self.num_subimages

    def __getitem__(self, idx):
        if isinstance(idx, tuple) and len(idx) == 3:
            return self.get_subimage(*idx)
        else:
            raise TypeError("Index must be a tuple of (channel, slice, subimage)")

    def create_subimages(self):
        """Split each image into subimages and store them in the subimages array."""
        for channel in range(self.num_channels):
            for slice_ in range(self.num_slices):
                for subimg in range(self.num_subimages):
                    start_row = subimg * self.subimage_height
                    end_row = (subimg + 1) * self.subimage_height
                    self.data[channel, slice_, subimg, :, :] = self.original_data[channel, slice_, start_row:end_row, :]

    def get_subimage(self, channel_index, slice_index, subimage_index):
        """Return a specific subimage from the dataset."""
        if channel_index < 0 or channel_index >= self.num_channels or \
           slice_index < 0 or slice_index >= self.num_slices or \
           subimage_index < 0 or subimage_index >= self.num_subimages:
            raise ValueError("Invalid channel, slice, or subimage index")
        return self.data[channel_index, slice_index, subimage_index, :, :]

    def show_subimage(self, channel_index, slice_index, subimage_index):
        """Display a specific subimage."""
        subimage = self.get_subimage(channel_index, slice_index, subimage_index)
        plt.imshow(subimage, cmap='gray')
        plt.title(f"Channel {channel_index}, Slice {slice_index}, Subimage {subimage_index}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    # Create a dataset object
    my_filename = "data/4-SIM-merge-18frames.tif"
    dataset = SIMDataset(my_filename)

    # Print some information about the dataset
    print(dataset)
    print(f"Number of subimages: {len(dataset)}")

    # Show a specific subimage
    dataset.show_subimage(0, 0, 0)
