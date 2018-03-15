from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import pandas as pd


class CustomDatasetFromImages(Dataset):
    def __init__(self, csv_path):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=None)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        # # Third column is for an operation indicator
        # self.operation_arr = np.asarray(self.data_info.iloc[:, 2])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)

        # # Check if there is an operation
        # some_operation = self.operation_arr[index]
        # # If there is an operation
        # if some_operation:
        #     # Do some operation on image
        #     # ...
        #     # ...
        #     pass
        # Transform image to tensor
        img_as_tensor = self.to_tensor(img_as_img)
        return img_as_tensor

    def __len__(self):
        return self.data_len


if __name__ == "__main__":
    """"The following does not use a DataLoader."""
    # Call dataset
    ultrasound_dataset_from_images =  \
        CustomDatasetFromImages('ultrasound_images.csv')

    for i in range(len(ultrasound_dataset_from_images)):
        sample = ultrasound_dataset_from_images[i]
        print(i, sample.size())
        if i == 3:
            break

if __name__ == "__main__":
    """The following uses DataLoader, which helps for organising batches."""
    # Define custom dataset
    ultrasound_dataset_from_images = \
        CustomDatasetFromImages('ultrasound_images.csv')
    # Define data loader
    mn_dataset_loader = DataLoader(dataset=ultrasound_dataset_from_images,
                                                    batch_size=10,
                                                    shuffle=False)

    for images in mn_dataset_loader:
# # Feed the data to the model

