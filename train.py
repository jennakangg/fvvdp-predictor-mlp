import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.lod_mlp_model import LowestLODPredictor  # Ensure this is the correct import path for your model
import json
from dataset.lod_fvvdp_eccentricity import LODFvvdpEccentricity
from torch.utils.data import DataLoader, random_split, Subset
import wandb
from data.preprocess import prepare_inputs_and_targets
from wandb_logging import log_gradients
import os

def load_data(config):
    # Set the random seed for reproducibility
    torch.manual_seed(config['seed'])

    # Initialize the dataset
    dataset = LODFvvdpEccentricity("dataset/playroom_lod_ecc", "dataset")

    train_size = int(len(dataset) * config['train_split'])
    valid_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=prepare_inputs_and_targets)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=prepare_inputs_and_targets)

    return train_loader, valid_loader

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, batch in enumerate(train_loader):
            inputs, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # log_gradients(model, epoch, idx, train_loader)

            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        wandb.log({'epoch': epoch, 'train_loss': avg_train_loss})
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}')


        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                inputs, targets = batch
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(valid_loader)
        wandb.log({'epoch': epoch, 'val_loss': avg_val_loss})
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}')

def save_checkpoint(model, epoch, path='checkpoints/model_checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    wandb.init(project='LODMLPPredictor', config=config)
    os.environ["WANDB_SILENT"] = "true"

    config["device"]= "cpu" if not torch.cuda.device_count() else "cuda:0"

    # Initialize the model
    model = LowestLODPredictor(input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_dim=config['hidden_dim'])
    model = model.to(config["device"])
    # def init_weights(m):
    #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    # model.apply(init_weights)

    # Load data
    train_loader, valid_loader = load_data(config)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error Loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    # Train the model
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=config['epochs'])

    # Save the model checkpoint
    save_checkpoint(model, config['epochs'])

if __name__ == '__main__':
    main()
