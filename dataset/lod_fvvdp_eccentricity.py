import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class LODFvvdpEccentricity(Dataset):
    def __init__(self, dataset_dir, root_dir):
        """
        Initialize the dataset by loading the metadata and setting the dataset directory.
        """
        self.dataset_dir = dataset_dir
        self.root_dir = root_dir
        
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=['camera_position', 'eccentricity', 'theta', 'camera_dir',
                                                  'lod_n_path', 'lod_x_path', 'lod_n', 'lod_x', 'view_index',
                                                  'pixels_per_deg', 'levels', 'heatmap_path', 'JOD_average'])
        
        self.ecc_mean, self.ecc_std, self.theta_mean, self.theta_std = self.compute_statistics()

    def __len__(self):
        """
        Return the total number of patches in the dataset.
        """
        return len(self.metadata)
    
    def compute_statistics(self):
        """
        Compute and return the mean and standard deviation for eccentricity and theta.
        """
        if 'eccentricity' in self.metadata.columns and 'theta' in self.metadata.columns:
            ecc_mean = self.metadata['eccentricity'].mean()
            ecc_std = self.metadata['eccentricity'].std()
            theta_mean = self.metadata['theta'].mean()
            theta_std = self.metadata['theta'].std()
            return ecc_mean, ecc_std, theta_mean, theta_std
        return 0, 1, 0, 1 
    
    def __getitem__(self, index):
        """
        Get an item by index and normalize 'eccentricity' and 'theta'.
        """
        meta = self.metadata.iloc[index]
        heatmap_path = os.path.join(self.root_dir, meta['heatmap_path'])
        lod_n_path = os.path.join(self.root_dir, meta['lod_n_path'])
        lod_x_path = os.path.join(self.root_dir, meta['lod_x_path'])
        heatmap_patch = torch.load(heatmap_path)
        lod_n_patch = torch.load(lod_n_path)
        lod_x_patch = torch.load(lod_x_path)

        normalized_eccentricity = meta['eccentricity']/self.ecc_mean
        normalized_theta = meta['theta']/self.theta_mean

        sample = {
            'camera_position': meta['camera_position'],
            'eccentricity': normalized_eccentricity,
            'theta': normalized_theta,
            'lod_n_patch': lod_n_patch,
            'lod_x_patch': lod_x_patch,
            'lod_n': meta['lod_n'],
            'lod_x': meta['lod_x'],
            'camera_dir': meta['camera_dir'],
            'view_index': meta['view_index'],
            'pixels_per_deg': meta['pixels_per_deg'],
            'levels': meta['levels'],
            'heatmap': heatmap_patch,
            'JOD_average': meta['JOD_average']
        }

        return sample
