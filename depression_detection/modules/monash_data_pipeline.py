import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence



class DepressionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



class GeneralizationTestingDataPipeline:
    def __init__(
        self,
        au_dir: str,
        mfcc_dir: str, 
        keep_AU_cols: list,
        daic_woz_train_split_df: pd.DataFrame = None, 
        au_separate: bool = True, 
        au_fixed_length: int = None,
        mfcc_fixed_length: int = None,
        daic_woz_mfcc_dir: str = None,
        segment_config: dict = None,
        batch_size: int = 32, 
    ):
        """
        Note:
            - The following code is hard coded to follow the specifications of the DAIC-WOZ AU-MFCC model
            - It's function is only to be used for testing the models' generalizability on the Monash dataset.

        Args:
            - au_dir (str): Directory containing au files.
            - mfcc_dir (str): Directory containing MFCC .npy files.
            - keep_AU_cols (list): List of strings containing AU columns to keep

            - daic_woz_split_df (pd.Dataframe, optional): Dictionary containing training split data from DAIC-WOZ.
            
            - au_separate (bool): If True, keep AU_r and AU_c separate; if False, concatenate them along the feature axis.
            - au_fixed_length (int, optional): If provided, all AU sequences are forced to this length.

            - mfcc_fixed_length (int, optional): If provided, all MFCC sequences are forced to this length.
            - daic_woz_mfcc_dir (str) = Directory containing DAIC-WOZ MFCC .npy files.

            - segment_config (dict, optional): Dictionary with keys for segmenting the time series data:
                - "au_window_size", "au_step_size"
                - "mfcc_window_size", "mfcc_step_size"
                When provided, each segmentation window (for AU and/or MFCC) produces a new sample.
            
            - batch_size (int): Batch size for the dataloaders.
        """
        # Variables
        self.au_separate = au_separate
        self.au_fixed_length = au_fixed_length
        self.mfcc_fixed_length = mfcc_fixed_length
        self.batch_size = batch_size

        # Prepare the data splits.
        self.data_splits = self.prepare_data(daic_woz_train_split_df, daic_woz_mfcc_dir, au_dir, mfcc_dir, keep_AU_cols, segment_config)

        # Create dataloaders.
        self.dataloaders = self.create_dataloaders(self.data_splits)


    def prepare_data(self, daic_woz_split_df: pd.DataFrame, daic_woz_mfcc_dir: str, au_dir: str, mfcc_dir: str, keep_AU_cols: list, segment_config: dict):
        """
        Prepares the AU and MFCC data for train/val/test splits.
        
        Depending on whether au_df and mfcc_dir are provided, the final sample will include:
          - For AU: Either separate "AU_r" and "AU_c" (or combined "AUs").
          - MFCCs.
          - Optionally, gender.
          - Always, the category from the split_df.
        
        When segment_config is provided, instead of storing segmented windows in a single sample,
        each valid window becomes its own sample (with all non-time-series data copied over).
        
        Returns:
           - dict: A dictionary with keys 'train', 'val', and 'test', each containing a list of sample dictionaries.
        """
        print("Preparing Data")

        # Normalize MFCCs on DAIC-WOZ training data if provided
        if mfcc_dir is not None and daic_woz_mfcc_dir is not None:
            mfcc_scaler = StandardScaler()
            train_ids = daic_woz_split_df['Participant_ID']
            train_mfccs = np.vstack([
                np.load(os.path.join(daic_woz_mfcc_dir, f"{pid}_P/{pid}_MFCC.npy")).T
                for pid in train_ids
            ])
            mfcc_scaler.fit(train_mfccs)


        data_splits = {'test': []}

        
        # For each participant
        for i in tqdm(range(1,8), desc="TEST"):
            # For each severity
            for severity in ["200(Mild)", "300(Moderate)", "400(Severe)"]:
                sample = {}

                # Load AU data
                au_file_path = os.path.join(au_dir, f"S0{i}/{severity}/S0{i}_{severity[:3]}.csv")
                sample_au = pd.read_csv(au_file_path).iloc[::4, :] # 120fps -> 30fps
                sample_au = sample_au[sample_au["success"] == 1]
                sample_au = sample_au[keep_AU_cols]
    
                # Separate regression and classification AUs if specified
                if self.au_separate:
                    AU_cols_cont = sample_au.filter(regex="AU\d+_r").columns
                    AU_cols_bin = sample_au.filter(regex="AU\d+_c").columns

                    if not AU_cols_cont.empty:
                        sample['AU_r'] = sample_au[AU_cols_cont].to_numpy() 

                    if not AU_cols_bin.empty:
                        sample['AU_c'] = sample_au[AU_cols_bin].to_numpy() 

                # Else load all AUs together
                else:
                    AU_cols = sample_au.filter(regex="AU*").columns
                    sample['AUs'] = sample_au[AU_cols].to_numpy()          

                # Load and process (normalize) MFCC data
                mfcc_path = os.path.join(mfcc_dir, f"S0{i}/{severity}/S0{i}_{severity[:3]}.npy")
                mfcc_data = np.load(mfcc_path).T                                        # Shape: (num_frames, coefficients)
                if daic_woz_mfcc_dir is not None:
                    mfcc_data = mfcc_scaler.transform(mfcc_data)
                sample['MFCCs'] = mfcc_data

                # Add Category
                category = 0 if severity == "200(Mild)" else 1
                sample['Category'] = category

                # If segmentation is enabled, break the sample into multiple segmented samples.
                if segment_config is not None:
                    segmented_samples = self._segment_sample(sample, segment_config)
                    data_splits["test"].extend(segmented_samples)
                else:
                    data_splits["test"].append(sample)

        return data_splits


    def _segment_sample(self, sample: dict, segment_config: dict) -> list:
        """
        Segments the time-series data in a sample into multiple samples based on the provided configuration.
        Each segment of AU and/or MFCC becomes a new sample, while other fields remain unchanged.
        
        Args:
            - sample (dict): Original sample containing time-series data (e.g., 'AU_r', 'AU_c', 'AUs', 'MFCCs').
            - segment_config (dict): Dictionary with segmentation parameters:
                - For AU: "au_window_size", "au_step_size"
                - For MFCC: "mfcc_window_size", "mfcc_step_size"
                
        Returns:
            list: A list of segmented sample dictionaries.
        """
        segments_dict = {}
        num_segments = None

        # Segment AU data if available.
        for AU_type in ['AU_r', 'AU_c', 'AUs']:
            if AU_type in sample:
                segments_au = self._segment_data(sample[AU_type],
                                                 segment_config.get("au_window_size"),
                                                 segment_config.get("au_step_size"))
            
                if num_segments is None:
                    num_segments = len(segments_au)

                segments_dict[AU_type] = segments_au[:num_segments]

        # Segment MFCC data if available.
        if 'MFCCs' in sample:
            segments_mfcc = self._segment_data(sample['MFCCs'],
                                               segment_config.get("mfcc_window_size"),
                                               segment_config.get("mfcc_step_size"))
            if num_segments is None:
                num_segments = len(segments_mfcc)
            else:
                num_segments = min(num_segments, len(segments_mfcc))
            segments_dict['MFCCs'] = segments_mfcc

        # If no segmentation was performed, return the original sample.
        if num_segments is None or num_segments == 0:
            return [sample]

        # Create new samples for each segmentation window.
        segmented_samples = []
        for i in range(num_segments):
            new_sample = sample.copy()  # shallow copy for other fields
            if 'AU_r' in segments_dict and 'AU_c' in segments_dict:
                new_sample['AU_r'] = segments_dict['AU_r'][i]
                new_sample['AU_c'] = segments_dict['AU_c'][i]
            elif 'AUs' in segments_dict:
                new_sample['AUs'] = segments_dict['AUs'][i]
            if 'MFCCs' in segments_dict:
                new_sample['MFCCs'] = segments_dict['MFCCs'][i]
            segmented_samples.append(new_sample)

        return segmented_samples


    def _segment_data(self, data: np.ndarray, window_size: int, step_size: int):
        """
        Segments time-series data using a sliding window.
        
        Args:
            - data (np.ndarray): Time-series data (shape: num_frames, num_features).
            - window_size (int): Number of frames per segment.
            - step_size (int): Step size for the sliding window.
        
        Returns:
            list: List of segments, each of shape (window_size, num_features).
        """
        segments = [
            data[i: i + window_size]
            for i in range(0, len(data) - window_size + 1, step_size)
        ]
        return segments


    def create_dataloaders(self, data):
        """
        Creates PyTorch dataloaders for each data split.
        
        The collate function forces every sequence to a fixed length (if specified) while preserving the original lengths.
        For the training split, if weighted_random_sampling is provided, a WeightedRandomSampler is used.
        """
        def fixed_length_tensor(tensor: torch.Tensor, fixed_length: int):
            current_length = tensor.shape[0]
            if current_length >= fixed_length:
                return tensor[:fixed_length], fixed_length
            else:
                pad_size = fixed_length - current_length
                pad_tensor = torch.zeros(pad_size, tensor.shape[1], dtype=tensor.dtype)
                return torch.cat([tensor, pad_tensor], dim=0), current_length

        def collate_fn(batch):
            out = {}
            # Process AU features 
            for AU_type in ['AU_r', 'AU_c', 'AUs']:
                if AU_type in batch[0]:
                    AU_list = [torch.tensor(sample[AU_type], dtype=torch.float32) for sample in batch]
                    if self.au_fixed_length is not None:
                        AU_list, prepadded_lengths = zip(*[fixed_length_tensor(t, self.au_fixed_length) for t in AU_list])
                        out[AU_type] = torch.stack(AU_list)
                        out["AU_lengths"] = prepadded_lengths
                    else:
                        out[AU_type] = pad_sequence(AU_list, batch_first=True)
                        out["AU_lengths"] = torch.tensor([t.shape[0] for t in AU_list], dtype=torch.long)

            # Process MFCC features.
            if 'MFCCs' in batch[0]:
                MFCCs_list = [torch.tensor(sample['MFCCs'], dtype=torch.float32) for sample in batch]
                if self.mfcc_fixed_length is not None:
                    MFCCs_list, prepadded_lengths = zip(*[fixed_length_tensor(t, self.mfcc_fixed_length) for t in MFCCs_list])
                    out['MFCCs'] = torch.stack(MFCCs_list)
                    out["MFCCs_lengths"] = prepadded_lengths
                else:
                    out['MFCCs'] = pad_sequence(MFCCs_list, batch_first=True)
                    out["MFCCs_lengths"]= torch.tensor([t.shape[0] for t in MFCCs_list], dtype=torch.long)

            # Process Category.
            categories = [sample['Category'] for sample in batch]
            out['Category'] = torch.tensor(categories, dtype=torch.long)

            return out

        dataloaders = {}
        dataset = DepressionDataset(data["test"])
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        dataloaders["test"] = dataloader

        return dataloaders


    def dataset_sample_summary(self, split: str, sample_no: int):
        """
        Prints summary for a given sample from a specified split.
        """
        sample = self.dataloaders[split].dataset[sample_no]
        for key, value in sample.items():
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                print(f'{key} Shape: {value.shape}')
            else:
                print(f'{key}: {value}')
