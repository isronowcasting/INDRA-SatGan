from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import numpy as np
import torch
from pytorch_lightning import LightningDataModule
import os
import pandas as pd

class HDF5Dataset(Dataset):
    """
    A simple dataset class that receives a list of pre-defined sequences
    and loads the corresponding HDF5 files.
    """
    def __init__(self, data_dir, sequence_list, num_input_frames, num_target_frames):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sequences = sequence_list
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence_files = self.sequences[index]
        input_files = sequence_files[:self.num_input_frames]
        target_files = sequence_files[self.num_input_frames:]

        input_frames = []
        for file in input_files:
            with h5py.File(os.path.join(self.data_dir, file), "r") as f:
                input_frames.append(f["precipitationCal"][:])

        target_frames = []
        for file in target_files:
            with h5py.File(os.path.join(self.data_dir, file), "r") as f:
                target_frames.append(f["precipitationCal"][:])
        
        input_tensor = torch.tensor(np.array(input_frames), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(target_frames), dtype=torch.float32)
        
        return input_tensor, target_tensor


class DGMRDataModule(LightningDataModule):
    """
    The LightningDataModule is responsible for all data handling.
    """
    def __init__(self, dataset_folder, csv_path, val_split=0.05, num_workers=50, pin_memory=True, batch_size=16, num_input_frames=6, num_target_frames=6):
        super().__init__()
        self.dataset_folder = dataset_folder
        self.csv_path = csv_path
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.batch_size = batch_size
        self.num_input_frames = num_input_frames
        self.num_target_frames = num_target_frames
        self.train_sequences = None
        self.val_sequences = None
        self.has_logged_summary = False

    def setup(self, stage=None):
        if self.train_sequences is not None and self.val_sequences is not None:
            return

        all_files_df = pd.read_csv(self.csv_path, header=None)
        
        all_sequences = []
        num_frames_per_sequence = self.num_input_frames + self.num_target_frames

        for index, row in all_files_df.iterrows():
            file_list = row.dropna().tolist()
            if len(file_list) >= num_frames_per_sequence:
                for i in range(len(file_list) - num_frames_per_sequence + 1):
                    all_sequences.append(file_list[i : i + num_frames_per_sequence])

        if not all_sequences:
            raise ValueError(f"No valid sequences generated. Check CSV and frame counts (need {num_frames_per_sequence} frames).")

        np.random.shuffle(all_sequences)
        split_idx = int(len(all_sequences) * (1 - self.val_split))
        self.train_sequences = all_sequences[:split_idx]
        self.val_sequences = all_sequences[split_idx:]
        
        if not self.has_logged_summary:
            self.log_summary()
            self.has_logged_summary = True

    def log_summary(self):
        """Calculates and prints a detailed summary of the dataset splits."""
        print("\n" + "="*50)
        print(" " * 15 + "DATASET SUMMARY")
        print("="*50)

        all_files_in_csv = pd.read_csv(self.csv_path, header=None).values.flatten()
        unique_files_in_csv = len(pd.unique(all_files_in_csv[~pd.isna(all_files_in_csv)]))
        total_sequences = len(self.train_sequences) + len(self.val_sequences)
        
        print("\n[Overall Data Source]")
        print(f"  Total unique HDF5 files found in CSV: {unique_files_in_csv}")
        print(f"  Total sliding-window sequences generated: {total_sequences}")

        train_files = set(f for seq in self.train_sequences for f in seq)
        print("\n[Training Set]")
        print(f"  Number of sequences: {len(self.train_sequences)}")
        print(f"  Number of unique files: {len(train_files)}")
        print(f"  Batch size: {self.batch_size}")
        
        # --- CORRECTED LOGIC HERE ---
        if len(self.train_sequences) > 0:
            print(f"  Batches per epoch: {len(self.train_dataloader())}")
        else:
            print("  Batches per epoch: 0")
        # ----------------------------
        
        val_files = set(f for seq in self.val_sequences for f in seq)
        print("\n[Validation Set]")
        print(f"  Number of sequences: {len(self.val_sequences)}")
        print(f"  Number of unique files: {len(val_files)}")
        print(f"  Batch size: {self.batch_size}")
        
        # --- CORRECTED LOGIC HERE ---
        if len(self.val_sequences) > 0:
            print(f"  Batches per epoch: {len(self.val_dataloader())}")
        else:
            print("  Batches per epoch: 0")
        # ----------------------------
        
        print("="*50 + "\n")

    def train_dataloader(self):
        if self.train_sequences is None: self.setup()
        # --- ADDED SAFETY CHECK ---
        if not self.train_sequences:
            # Return an empty dataloader if there are no sequences
            return DataLoader(HDF5Dataset(self.dataset_folder, [], self.num_input_frames, self.num_target_frames), batch_size=self.batch_size)
        # --------------------------
        dataset = HDF5Dataset(self.dataset_folder, self.train_sequences, self.num_input_frames, self.num_target_frames)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        if self.val_sequences is None: self.setup()
        dataset = HDF5Dataset(self.dataset_folder, self.val_sequences, self.num_input_frames, self.num_target_frames)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)