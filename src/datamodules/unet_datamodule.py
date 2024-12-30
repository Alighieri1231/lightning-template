import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torchvision.transforms import functional as F
import torchio as tio


class SegmentationDataset(Dataset):
    def __init__(self, file_paths, data_path, transform=None):
        """
        Args:
            file_paths (list of str): List of .npz file names.
            data_path (str): Path to the directory containing the .npz files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.file_paths = file_paths
        self.data_path = Path(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.data_path / self.file_paths[idx]
        data = np.load(file_path)
        video = data["video"]  # Shape: (256, 512, 512)
        label = data["label"]  # Shape: (256, 512, 512)

        subject = tio.Subject(
            video_gt=tio.ScalarImage(
                tensor=np.expand_dims(video, axis=0)
            ),  # Add channel dimension
            label=tio.LabelMap(
                tensor=np.expand_dims(label, axis=0)
            ),  # Add channel dimension
        )

        # efine the resize transform
        resize_transform = tio.transforms.Resize(
            (128,128,128)
        )  # New shape: [frames, width, height]

        # Apply the resize transform to the subject
        resized_subject = resize_transform(subject)

        # Extract resized video GT and label from the subject
        video = resized_subject.video_gt.tensor  # Remove channel dimension
        label = resized_subject.label.tensor.squeeze(0).long()  # Ensure integers for label

        # Normalize video values to [0, 1]
        video = video.float() / 255.0

        # Convert to PyTorch tensors
        #video = torch.from_numpy(video).unsqueeze(0)  # Add channel dimension
        #label = torch.from_numpy(label).long()
        #print(video.shape)
        #print(label.shape)

        # Apply transform if provided
        if self.transform:
            video, label = self.transform((video, label))

        return video, label


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self, data_path, train_csv, val_csv, test_csv, batch_size=4, num_workers=4
    ):
        """
        Args:
            data_path (str): Path to the directory containing .npz files.
            train_csv (str): Path to the CSV file listing training .npz file names.
            val_csv (str): Path to the CSV file listing validation .npz file names.
            test_csv (str): Path to the CSV file listing testing .npz file names.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of workers for dataloaders.
        """
        super().__init__()
        self.data_path = data_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """
        Load datasets based on CSV files.
        Args:
            stage (str, optional): Stage of the setup (fit, validate, test, predict).
        """
        train_files = pd.read_csv(self.train_csv)["file_name"].tolist()
        val_files = pd.read_csv(self.val_csv)["file_name"].tolist()
        test_files = pd.read_csv(self.test_csv)["file_name"].tolist()

        self.train_dataset = SegmentationDataset(train_files, self.data_path)
        self.val_dataset = SegmentationDataset(val_files, self.data_path)
        self.test_dataset = SegmentationDataset(test_files, self.data_path)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


# Example usage:
# datamodule = SegmentationDataModule(
#     data_path="data_directory",
#     train_csv="train.csv",
#     val_csv="val.csv",
#     test_csv="test.csv",
#     batch_size=8,
#     num_workers=4
# )
# datamodule.setup()
# train_loader = datamodule.train_dataloader()
