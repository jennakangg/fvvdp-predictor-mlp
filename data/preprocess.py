import torch
import re

def list_to_tensor_3d(string, device):
    string = ''.join(string)  # Ensure it's a single string without any list brackets
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", string)  # Extract numbers

    numbers = [float(num) for num in numbers]    
    numbers = numbers[:3]
    tensor = torch.tensor(numbers, dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)  # Shape it as [1, N] where N is number of features (e.g., 3 for XYZ coordinates)

def normalize(tensor, mean, std, device):
    """Normalize a tensor using the specified mean and standard deviation."""
    return (tensor - mean) / std.to(device=device)

def prepare_inputs_and_targets(batch):
    """
    Prepare the input tensors and target values from the batch.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    inputs = []
    targets = []
    for item in batch:
        cam_pos = list_to_tensor_3d(item[0]['camera_position'], device=device)
        cam_pos = cam_pos/torch.norm(cam_pos)
        # ray_dir = torch.tensor(item['ray_dir'], device=device).unsqueeze(0)
        ray_dir = list_to_tensor_3d(item[0]['ray_dir'], device=device)
        
        eccentricity = torch.tensor([item[0]['eccentricity']], device=device).unsqueeze(0)
        theta = torch.tensor([item[0]['theta']], device=device).unsqueeze(0)

        JOD_array = []
        # should be JOD level 0, level 1, .... ->
        for level, lod_level_entry in enumerate(reversed(item)):
            JOD_array.append(lod_level_entry['LOD_value_JOD_less_than_1'])


        input_tensor = torch.cat([cam_pos, ray_dir, eccentricity, theta], dim=1)
        inputs.append(input_tensor)

        JOD_input = torch.tensor([JOD_array], dtype=torch.float32, device=device)
        targets.append(JOD_input)  

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)


    return inputs, targets

def prepare_inputs_and_targets_predict_JOD_1(batch):
    """
    Prepare the input tensors and target values from the batch.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    inputs = []
    targets = []
    for item in batch:
        cam_pos = list_to_tensor_3d(item[0]['camera_position'], device=device)
        cam_pos = cam_pos/torch.norm(cam_pos)
        # ray_dir = torch.tensor(item['ray_dir'], device=device).unsqueeze(0)
        ray_dir = list_to_tensor_3d(item[0]['ray_dir'], device=device)
        
        eccentricity = torch.tensor([item[0]['eccentricity']], device=device).unsqueeze(0)
        theta = torch.tensor([item[0]['theta']], device=device).unsqueeze(0)


        input_tensor = torch.cat([cam_pos, ray_dir, eccentricity, theta], dim=1)
        inputs.append(input_tensor)

        JOD_input = torch.tensor([item[0]['LOD_value_JOD_less_than_1']], dtype=torch.float32, device=device).unsqueeze(0)
        targets.append(JOD_input)  

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)


    return inputs, targets