import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class CameraParametersDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, 'camera_metadata.csv')

        if os.path.exists(metadata_path):
            self.metadata = pd.read_csv(metadata_path)
        else:
            self.metadata = pd.DataFrame(columns=['camera_position', 'view_idx', 'camera_dir', 'camera_R_path', 'camera_T_path'])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        meta = self.metadata.iloc[index]
        camera_R = torch.load(meta['camera_R_path'])
        camera_T = torch.load(meta['camera_T_path'])

        return {
            'camera_position': meta['camera_position'],
            'view_idx': meta['view_idx'],
            'camera_dir': meta['camera_dir'],
            'camera_R': camera_R,
            'camera_T': camera_T
        }

    def add_entry(self, camera_position, camera_dir, camera_R, camera_T, view_idx):
        base_filename = f"v_{view_idx}"
        camera_R_path = os.path.join(self.dataset_dir, f"{base_filename}_R.pt")
        camera_T_path = os.path.join(self.dataset_dir, f"{base_filename}_T.pt")

        torch.save(camera_R, camera_R_path)
        torch.save(camera_T, camera_T_path)

        new_entry = {
            'camera_position': camera_position,
            'view_idx': view_idx,
            'camera_dir': camera_dir,
            'camera_R_path': camera_R_path,
            'camera_T_path': camera_T_path
        }

        new_entry_df = pd.DataFrame([new_entry])
        self.metadata = pd.concat([self.metadata, new_entry_df], ignore_index=True)

        metadata_path = os.path.join(self.dataset_dir, 'camera_metadata.csv')
        self.metadata.to_csv(metadata_path, index=False)

    def get_entry_by_position(self, camera_position):
        matches = self.metadata[self.metadata['camera_position'] == camera_position]
        if not matches.empty:
            meta = matches.iloc[0]
            camera_R = torch.load(meta['camera_R_path'])
            camera_T = torch.load(meta['camera_T_path'])

            return {
                'camera_position': meta['camera_position'],
                'view_idx': meta['view_idx'],
                'camera_dir': meta['camera_dir'],
                'camera_R': camera_R,
                'camera_T': camera_T
            }
        return None
