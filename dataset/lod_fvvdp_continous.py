import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import re
import numpy as np
from collections import defaultdict

class LODFvvdpEccentricityContinous(Dataset):
    def __init__(self, dataset_dir, root_dir, camera_position=None, target_ray_dir=None, epsilon=1e-2):
        self.dataset_dir = dataset_dir
        self.root_dir = root_dir
        metadata_path = os.path.join(dataset_dir, 'metadata.csv')
        
        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=['camera_position', 'eccentricity', 'theta', 'ray_dir',
                                                  'lod_n_path', 'lod_x_path', 'lod_n', 'lod_x', 'view_index',
                                                  'levels', 'heatmap_path', 'JOD_average', 'camera_R', 'camera_T'])
        
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

            # Transform ray directions to align with the y-axis
            self.metadata['transformed_ray_dir'] = self.metadata['ray_dir'].apply(
                lambda x: R.dot(x))

            # Filter for y-axis rotation
            self.metadata = self.metadata[self.metadata['transformed_ray_dir'].apply(
                lambda x: abs(x[0]) < epsilon and abs(x[2]) < epsilon and abs(x[1]) > epsilon)]

        self.groups = defaultdict(list)
        for index, row in self.metadata.iterrows():
            key = (row['camera_position'], row['eccentricity'], row['theta'])
            self.groups[key].append(index)

        # Ensure unique grouping keys
        self.group_keys = list(self.groups.keys())

    def __len__(self):
        """
        Return the total number of groups in the dataset.
        """
        return len(self.group_keys)

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
        Get a group by index.
        """
        if index >= len(self.group_keys):
            raise IndexError(f"Index {index} is out of bounds for dataset with {len(self.group_keys)} groups.")
        
        key = self.group_keys[index]
        indices = self.groups[key]

        def get_item(meta):
            heatmap_path = os.path.join(self.root_dir, meta['heatmap_path'])
            lod_n_path = os.path.join(self.root_dir, meta['lod_n_path'])
            lod_x_path = os.path.join(self.root_dir, meta['lod_x_path'])
            camera_R_path = os.path.join(self.root_dir, meta['camera_R'])
            camera_T_path = os.path.join(self.root_dir, meta['camera_T'])
            heatmap_patch = torch.load(heatmap_path)
            lod_n_patch = torch.load(lod_n_path)
            lod_x_patch = torch.load(lod_x_path)
            camera_R = torch.load(camera_R_path)
            camera_T = torch.load(camera_T_path)

            normalized_eccentricity = float(meta['eccentricity']) / 27
            normalized_theta = float(meta['theta']) / (2 * np.pi)

            return {
                'camera_position': meta['camera_position'],
                'eccentricity': normalized_eccentricity,
                'camera_dir': meta['camera_dir'],
                'camera_R': camera_R,
                'camera_T': camera_T,
                'theta': normalized_theta,
                'lod_n_patch': lod_n_patch,
                'lod_x_patch': lod_x_patch,
                'lod_n': meta['lod_n'],
                'lod_x': meta['lod_x'],
                'ray_dir': meta['ray_dir'],
                'view_index': meta['view_index'],
                'levels': meta['levels'],
                'heatmap': heatmap_patch,
                'JOD_average': meta['JOD_average']
            }

        group = [get_item(self.metadata.iloc[i]) for i in indices]
        return group

    def add_entry(self, entry_data):
        """
        Save new data entry into the dataset.
        
        entry_data should include:
        - camera_position, eccentricity, ray_dir, lod_n_patch, lod_x_patch,
          lod_n, lod_x, view_index, pixels_per_deg, levels, heatmap_patch
        """
        base_filename = f"v{entry_data['view_index']}_e{entry_data['eccentricity']}_t{entry_data['theta']}"
        heatmap_filename = os.path.join(self.dataset_dir, f"{base_filename}_heatmap.pt")
        lod_n_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_n.pt")
        lod_x_filename = os.path.join(self.dataset_dir, f"{base_filename}_lod_x.pt")

        torch.save(entry_data['lod_n_patch'], lod_n_filename)
        torch.save(entry_data['lod_x_patch'], lod_x_filename)
        torch.save(entry_data['heatmap_patch'], heatmap_filename)

        new_entry = {
            'camera_position': entry_data['camera_position'],
            'eccentricity': entry_data['eccentricity'],
            'ray_dir': entry_data['ray_dir'],
            'theta': entry_data['theta'],
            'lod_n_path': lod_n_filename,
            'lod_x_path': lod_x_filename,
            'lod_n': entry_data['lod_n'],
            'lod_x': entry_data['lod_x'],
            'view_index': entry_data['view_index'],
            'levels': entry_data['levels'],
            'heatmap_path': heatmap_filename,
            'JOD_average': entry_data['JOD_average']
        }
        new_entry_df = pd.DataFrame([new_entry]) 
        self.metadata = pd.concat([self.metadata, new_entry_df], ignore_index=True)

        # Update the grouping
        key = (new_entry['camera_position'], new_entry['ray_dir'])
        self.groups[key].append(len(self.metadata) - 1)
        if key not in self.group_keys:
            self.group_keys.append(key)

        metadata_path = os.path.join(self.dataset_dir, 'metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)

    def get_unique_camera_position(self):
        """
        Return a list of unique view indices from the dataset.
        """
        return self.metadata['camera_position'].unique().tolist()

    def get_all_by_view_index(self, view_index):
        """
        Get all dataset items that have the specified view index.
        """
        filtered_samples = []
        for i in range(len(self.metadata)):
            if self.metadata.iloc[i]['view_index'] == view_index:
                filtered_samples.append(self.__getitem__(i))
        return filtered_samples
