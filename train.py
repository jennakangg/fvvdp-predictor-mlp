import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.lod_mlp_model import LowestLODPredictor  

import json
from dataset.lod_fvvdp_continous import LODFvvdpEccentricityContinous
from torch.utils.data import DataLoader, random_split, Subset
import wandb
from data.preprocess import prepare_inputs_and_targets
from wandb_logging import log_gradients
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split


def load_data(config):
    torch.manual_seed(config['seed'])

    # Initialize the dataset
    dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all", "dataset")

    train_size = int(len(dataset) * config['train_split'])
    valid_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)

    return train_loader, valid_loader

def load_data_split_by_view(config):
    torch.manual_seed(config['seed'])

    dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all", "dataset")

    # Get unique camera positions
    unique_camera_positions = dataset.get_unique_camera_position()
    
    # Split camera positions into training and validation sets
    train_cam_pos, valid_cam_pos = train_test_split(unique_camera_positions, test_size=config['valid_split'], random_state=config['seed'])

    print(train_cam_pos)
    print(valid_cam_pos)
    train_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all", "dataset", camera_position=train_cam_pos)
    valid_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all", "dataset", camera_position=valid_cam_pos)


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)
 
    return train_loader, valid_loader

def load_data_1(config):
    torch.manual_seed(config['seed'])

    train_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all", "dataset")
    valid_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_5_levels_all_test", "dataset")


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)

    return train_loader, valid_loader

def load_data_subset(config, num_samples=None):
    torch.manual_seed(config['seed'])

    train_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white", "dataset")
    valid_dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white_test", "dataset")

    if num_samples:
        train_dataset = Subset(train_dataset, range(min(num_samples, len(train_dataset))))
        valid_dataset = Subset(valid_dataset, range(min(num_samples, len(valid_dataset))))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)

    return train_loader, valid_loader

def load_data_subset_filtered(config, camera_position, target_ray_dir, num_samples=None):
    torch.manual_seed(config['seed'])

    dataset = LODFvvdpEccentricityContinous("dataset/example_lod_ecc_white", "dataset", camera_position, target_ray_dir)

    train_size = int(len(dataset) * config['train_split'])
    valid_size = len(dataset) - train_size

    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)

    return train_loader, valid_loader

def custom_absolute_error(outputs, targets):
    with torch.no_grad():
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        errors = np.abs(targets - outputs)

        return np.mean(errors)
    
def scaled_tanh(x):
    return (x + 1) / 2

def find_first_index_le_one(array):
    """
    Returns the first index in each row where the value is less than or equal to 1.

    Parameters:
    - array (torch.Tensor): A 2D tensor.

    Returns:
    - List[int]: The first index in each row where the value is less than or equal to 1.
                 Returns -1 if no such value is found.
    """
    indices = []
    for row in array:
        index = next((i for i, v in enumerate(row) if v <= 1), -1)
        indices.append(index)
    return indices

def count_correct_predictions(outputs, targets):
    """
    Counts how many indices match between the outputs and targets based on the first index
    of values less than or equal to 1.

    Parameters:
    - outputs (torch.Tensor): The output predictions.
    - targets (torch.Tensor): The target labels.

    Returns:
    - int: The count of correct predictions where the first index matches.
    """
    output_indices = find_first_index_le_one(outputs)
    target_indices = find_first_index_le_one(targets)

    correct_count = sum(o == t for o, t in zip(output_indices, target_indices))
    return correct_count

def decreasing_loss(output):
    """
    Computes a penalty for non-decreasing sequences in the output.

    Parameters:
    - output (torch.Tensor): Predicted outputs, assumed to be of shape (batch_size, n_indices).

    Returns:
    - torch.Tensor: The computed loss.
    """
    penalty = torch.sum(torch.relu(output[:, :-1] - output[:, 1:]))
    return penalty

def custom_loss(predictions, targets, decreasing_weight=1.0, criterion=nn.MSELoss()):
    standard_loss = criterion(predictions, targets)
    decreasing_penalty = decreasing_loss(predictions)
    return standard_loss + decreasing_weight * decreasing_penalty

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=100, level_n=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_absolute_error_train = 0.0


        total_values = 0

        for idx, batch in enumerate(train_loader):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            predicted_lods = outputs * 4
            total_absolute_error_train += custom_absolute_error(predicted_lods, targets)

            loss = criterion(predicted_lods, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()* inputs.size(0)

            total_values += len(outputs)

            # if epoch % 10 == 0:
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print(f"Epoch {epoch}, Gradient of {name}: {param.grad.norm().item()}")


        if epoch % 20 == 0:
            print("OUTPUTS")
            print(predicted_lods)
            print("TARGETS")
            print(targets)
            
        avg_train_loss = running_loss / len(train_loader)
        avg_absolute_error_train = total_absolute_error_train / total_values

        wandb.log({'epoch': epoch, 'train_loss': avg_train_loss, 'train_absolute_error': avg_absolute_error_train, 
                   } )

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Train Absolute Error: {avg_absolute_error_train:.10f}%,')
        
        model.eval()
        running_val_loss = 0.0
        total_absolute_error_val = 0.0

        total_values_val = 0

        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                outputs = model(inputs) 
                predicted_lods = outputs * 4

                val_loss = criterion(predicted_lods, targets)

                running_val_loss += val_loss.item()* inputs.size(0)

                total_absolute_error_val += custom_absolute_error(predicted_lods, targets)
            
                total_values_val += len(outputs)

        avg_val_loss = running_val_loss / len(valid_loader)
        avg_absolute_error_val = total_absolute_error_val / total_values_val

        if epoch % 20 == 0:
            
            print("VAL OUTPUTS")
            print(predicted_lods)
            print("VAL TARGETS")
            print(targets)

        wandb.log({'epoch': epoch, 'val_loss': avg_val_loss, 'val_absolute_error': avg_absolute_error_val, 
                   })
        if epoch == num_epochs-1:
            print("predicted actual!!!")
            print(predicted_lods)
            print("TARGETS")
            print(targets)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}, Validation Absolute Error: {avg_absolute_error_val:.10f}%, ')

    

def save_checkpoint(model, epoch, path='checkpoints/model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)

    wandb.save(path)

    print("saved checkpoint")

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)

    import os
    wandb.init(project='LODMLPPredictor_HGS_all_level_prediction', config=config)
    os.environ["WANDB_SILENT"] = "true"

    config["device"]= "cpu" if not torch.cuda.device_count() else "cuda:0"

    # Initialize the model
    model = LowestLODPredictor(input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_dim=config['hidden_dim'])
    model = model.to(config["device"])

    train_loader, valid_loader = load_data(config)

    criterion = nn.MSELoss()  
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # train_loader, valid_loader = load_data_subset(config, num_samples=1000)

    # camera_positions = [
    #     "tensor([-1.1648, -2.2401, -1.3973])",
    #     "tensor([1.5673,  2.1147, -1.2336])",
    #     "tensor([-1.2577, -0.2352, -2.1779])",
    #     "tensor([-3.7984, -3.4955, -3.0159])",
    #     "tensor([-4.3671,  0.4505, -3.5952])",
    #     "tensor([-1.1677,  0.9305, -2.3789])",
    #     "tensor([0.4296, -1.0359, -2.2853])",
    #     "tensor([-1.1991, -1.1011, -0.7496])"
    # ]

    # train_loader, valid_loader = load_data_subset_filtered(config, camera_position=camera_positions, target_ray_dir=None)

    try:
        train_model(model, train_loader, valid_loader, criterion, optimizer, config['epochs'])

        print("Training interrupted.")
    finally:
        save_checkpoint(model, epoch=10, path=f'checkpoints/model_checkpoint.pth')

    wandb.finish()

if __name__ == '__main__':
    main()
