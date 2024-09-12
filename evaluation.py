import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
from models.lod_mlp_model import LowestLODPredictor  
def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    with open('config.json', 'r') as f:
        config = json.load(f)
    model = LowestLODPredictor(input_dim=config['input_dim'], output_dim=config['output_dim'], hidden_dim=config['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])

    epoch = checkpoint['epoch']

    return model, epoch

def get_eccentricity_and_theta(x, y, img_width, img_height, pixels_per_deg):

    center_x, center_y = img_width / 2, img_height / 2
    
    delta_x = x - center_x
    delta_y = y - center_y

    theta = np.arctan2(delta_y, delta_x)  # Returns angle in radians
    
    eccentricity_pixels = np.sqrt(delta_x**2 + delta_y**2)

    eccentricity_deg = eccentricity_pixels / pixels_per_deg

    return eccentricity_deg, theta

def pixel_to_ray(x, y, image_width, image_height, ppd, R):

    center_x = image_width / 2
    center_y = image_height / 2

    angle_x = (x - center_x) / ppd
    angle_y = (y - center_y) / ppd

    # Convert angles to radians
    angle_x_rad = np.radians(angle_x)
    angle_y_rad = np.radians(angle_y)

    z = np.cos(angle_x_rad) * np.cos(angle_y_rad)
    x = np.sin(angle_x_rad) * np.cos(angle_y_rad)
    y = np.sin(angle_y_rad)

    direction_vector = np.array([x, y, z])

    direction_vector_normalized = direction_vector / np.linalg.norm(direction_vector)

    direction_vector_world = np.dot(R, direction_vector_normalized)

    return torch.tensor(direction_vector_world)

model, epoch = load_model('checkpoints/model_checkpoint.pth')
def scaled_tanh(x):
    return (x + 1) / 2

def calculate_inputs(camera_pos, i, j, width, height, ppd, R):
    ecc, theta = get_eccentricity_and_theta(i, j, width, height, ppd)
    
    ray_dir = pixel_to_ray(i, j, width, height, ppd, R)
    
    return camera_pos, ray_dir, ecc, theta


def predict_lod(camera_pos, ray_dir, eccentricity, theta, level_n=3):
    inputs = torch.tensor([camera_pos + list(ray_dir) + [eccentricity] + [theta]])
    outputs = model(inputs)
    # scaled_tanh_outputs = scaled_tanh(outputs)
    predicted_lods = outputs * level_n
    
    import math
    return math.ceil(predicted_lods.item())

def draw_legend(draw, start_x, start_y, color_map):
    font = ImageFont.load_default()
    box_height = 20
    box_width = 40
    text_offset = 10
    
    for index, (lod, color) in enumerate(color_map.items()):
        draw.rectangle([start_x, start_y, start_x + box_width, start_y + box_height], fill=color)
        draw.text((start_x + box_width + text_offset, start_y), f'LOD {lod}', fill=(255, 255, 255), font=font)
        start_y += box_height + 5  
def assemble_image(width, height, camera_pos, ppd):
    extended_width = width + 100  
    full_image = Image.new('RGB', (extended_width, height))
    patch_size = 32

    color_map = {
        0: (255, 0, 0),    # Red
        1: (0, 255, 0),    # Green
        2: (0, 0, 255),    # Blue
        3: (255, 255, 0),  # Yellow
        4: (255, 165, 0),  # Orange
        5: (75, 0, 130)    # Indigo
    }

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            center_i = i + patch_size // 2
            center_j = j + patch_size // 2

            starting_R = torch.tensor([[ 0.99978029 ,-0.01962377  ,0.0073681 ],
                                    [ 0.0209174   ,0.95678337 ,-0.29004836],
                                    [-0.00135783  ,0.29013876  ,0.95698363],])

            camera_pos, ray_dir, eccentricity, theta = calculate_inputs(camera_pos, center_i, center_j, width, height, ppd, starting_R)
            lod = predict_lod(camera_pos, ray_dir, eccentricity, theta)

            patch_color = color_map.get(lod, (128, 128, 128)) 
            patch_img = Image.new('RGB', (patch_size, patch_size), patch_color)
            full_image.paste(patch_img, (i, j))

    draw = ImageDraw.Draw(full_image)
    draw_legend(draw, width + 10, 10, color_map)

    full_image.save('output_lod_map.png')
    full_image.show()

camera_pos = np.array([ 3.3766, -0.6779, -3.1177]) 
camera_pos = list(camera_pos/np.linalg.norm(camera_pos))

width = 1024  # Define the width of the image
height = 690  # Define the height of the image
ppd = 32 
assemble_image(width, height, camera_pos, ppd)