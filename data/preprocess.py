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
        cam_pos = list_to_tensor_3d(item['camera_position'], device=device)
        # cam_dir = list_to_tensor_3d(item['camera_dir'], device=device)
        eccentricity = torch.tensor([item['eccentricity']], device=device).unsqueeze(0)
        theta = torch.tensor([item['theta']], device=device).unsqueeze(0)

        input_tensor = torch.cat([cam_pos, eccentricity, theta], dim=1)
        inputs.append(input_tensor)

        lod_x = torch.tensor([item['lod_x']], dtype=torch.float32, device=device).unsqueeze(0)
        targets.append(lod_x)  

    inputs = torch.cat(inputs, dim=0)
    targets = torch.cat(targets, dim=0)


    return inputs, targets