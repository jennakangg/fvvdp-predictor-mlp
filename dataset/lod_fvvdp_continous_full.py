import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from camera_params import CameraParametersDataset   
import re
import numpy as np

class LODFvvdpEccentricityContinous(Dataset):
    def __init__(self, dataset_dir, camera_dir, root_dir, camera_position=None, target_ray_dir=None, epsilon = 1e-2):
        """
        Initialize the dataset by loading the metadata and setting the dataset directory.
        """
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')

        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=['camera_position', 'eccentricity', 'theta', 'ray_dir',
                                                  'lod_n_path', 'lod_x_path', 'lod_n', 'lod_x', 'image_name',
                                                  'levels', 'heatmap_path', 'LOD_value_JOD_less_than_1'])

        self.camera_dataset = CameraParametersDataset(camera_dir)

        if 'camera_position' in self.metadata.columns:
            self.metadata['camera_position'] = self.metadata['camera_position'].astype(str)
        if 'ray_dir' in self.metadata.columns:
            self.metadata['ray_dir'] = self.metadata['ray_dir'].astype(str)

        if camera_position is not None:
            if isinstance(camera_position, str):
                self.metadata = self.metadata[self.metadata['camera_position'].str.contains(camera_position, na=False)]
            elif isinstance(camera_position, list):
                camera_position_regex = '|'.join(map(re.escape, camera_position))  # Use re.escape to avoid regex errors
                self.metadata = self.metadata[self.metadata['camera_position'].str.contains(camera_position_regex, na=False)]

        if target_ray_dir is not None:
            self.metadata['ray_dir'] = self.metadata['ray_dir'].apply(lambda x: np.fromstring(x.strip("tensor([").rstrip("])"), sep=","))

            y_axis = np.array([0, 1, 0])
            v = np.cross(target_ray_dir, y_axis)
            s = np.linalg.norm(v)
            c = np.dot(target_ray_dir, y_axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2)) if s != 0 else np.eye(3)

            self.metadata['transformed_ray_dir'] = self.metadata['ray_dir'].apply(
                lambda x: R.dot(x))

            # Filter for y-axis rotation
            self.metadata = self.metadata[self.metadata['transformed_ray_dir'].apply(
                lambda x: abs(x[0]) < epsilon and abs(x[2]) < epsilon and abs(x[1]) > epsilon)]


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
        Get an item by index.
        """
        meta = self.metadata.iloc[index]

        heatmap_path = os.path.join(self.root_dir, meta['heatmap_path'])
        lod_n_path = os.path.join(self.root_dir, meta['lod_n_path'])
        lod_x_path = os.path.join(self.root_dir, meta['lod_x_path'])
        heatmap_patch = torch.load(heatmap_path)
        lod_n_patch = torch.load(lod_n_path)
        lod_x_patch = torch.load(lod_x_path)

        normalized_eccentricity = float(meta['eccentricity'])/27
        normalized_theta = float(meta['theta'])/(2 * np.pi)

        camera_data = self.camera_dataset.get_entry_by_position(meta['camera_position'])
        camera_dir = camera_data['camera_dir']
        camera_R = camera_data['camera_R']
        camera_T = camera_data['camera_T']

        # Create the sample dictionary
        sample = {
            'camera_position': meta['camera_position'],
            'camera_dir': camera_dir,
            'camera_R': camera_R,
            'camera_T': camera_T,
            'eccentricity': normalized_eccentricity,
            'theta': normalized_theta,
            'lod_n_patch': lod_n_patch,
            'lod_x_patch': lod_x_patch,
            'lod_n': meta['lod_n'],
            'lod_x': meta['lod_x'],
            'ray_dir': meta['ray_dir'],
            'image_name': meta['image_name'],
            'levels': meta['levels'],
            'heatmap': heatmap_patch,
            'LOD_value_JOD_less_than_1': meta['LOD_value_JOD_less_than_1']
        }

        return sample

    def add_entry(self, entry_data):
        """
        Save a new data entry into the dataset.

        entry_data should include:
        - camera_position, eccentricity, ray_dir, lod_n_patch, lod_x_patch,
          lod_n, lod_x, image_name, pixels_per_deg, levels, heatmap_patch
        """
        base_filename = f"v{entry_data['image_name']}_e{entry_data['eccentricity']}_t{entry_data['theta']}"
        heatmap_filename = os.path.join(self.dataset_dir, f"{base_filename}_heatmap.pt")
        lod_n_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_n.pt")
        lod_x_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_x.pt")

        # Save the torch tensors
        torch.save(entry_data['lod_n_patch'], lod_n_filename)
        torch.save(entry_data['lod_x_patch'], lod_x_filename)
        torch.save(entry_data['heatmap_patch'], heatmap_filename)

        if not self.camera_dataset.get_entry_by_position(entry_data['camera_position']):
            self.camera_dataset.add_entry(
                entry_data['camera_position'], 
                entry_data['camera_dir'], 
                entry_data['camera_R'], 
                entry_data['camera_T'], 
                entry_data['image_name'])

        # Update the metadata
        new_entry = {
            'camera_position': entry_data['camera_position'],
            'eccentricity': entry_data['eccentricity'],
            'theta': entry_data['theta'],
            'lod_n_path': lod_n_filename,
            'lod_x_path': lod_x_filename,
            'lod_n': entry_data['lod_n'],
            'lod_x': entry_data['lod_x'],
            'image_name': entry_data['image_name'],
            'levels': entry_data['levels'],
            'ray_dir': entry_data['ray_dir'],
            'heatmap_path': heatmap_filename,
            'LOD_value_JOD_less_than_1': entry_data['LOD_value_JOD_less_than_1']
        }
        new_entry_df = pd.DataFrame([new_entry])  
        self.metadata = pd.concat([self.metadata, new_entry_df], ignore_index=True)

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)

    def get_all_by_image_name(self, image_name):
        """
        Get all dataset items that have the specified view index.
        """
        filtered_samples = []
        for i in range(len(self.metadata)):
            if self.metadata.iloc[i]['image_name'] == image_name:
                filtered_samples.append(self.__getitem__(i))
        return filtered_samples
    
    def get_unique_camera_position(self):
        """
        Return a list of unique view indices from the dataset.
        """
        return self.metadata['camera_position'].unique().tolist()
    
