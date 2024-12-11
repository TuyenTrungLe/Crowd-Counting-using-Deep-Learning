import torch

import numpy as np
import torch.nn as nn

from PIL import Image
from torchvision import transforms

from CSRNet import CSRNet
from FeatureExtractor import FeatureExtractor
from CustomRegressionModel import CustomRegressionModel


CSRNET_CHECKPOINT_PATH = "checkpoints/weights_pschaus.npz"
REGRESSION_MODEL_CHECKPOINT_PATH = "checkpoints/checkpoint_regression_one_dense.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load():
    # Load CSRNet model
    CSRNet_model = CSRNet().to(device)

    # Load the weights and checkpoints
    CSRNet_checkpoint = np.load(CSRNET_CHECKPOINT_PATH)
    CSRNet_checkpoint = {layer: torch.from_numpy(mat) for layer, mat in CSRNet_checkpoint.items()}
    CSRNet_model.load_state_dict(CSRNet_checkpoint)

    # Create FeatureExtractor and freeze the layers
    feature_extractor = FeatureExtractor(CSRNet_model)
    for param in feature_extractor.parameters():
        param.requires_grad = False

    # Create Custom Regression model
    regression_model = CustomRegressionModel(feature_extractor).to(device)

    # Initialize Dense Layer with RandomNormal
    nn.init.normal_(regression_model.dense.weight, mean=1., std=0.001)
    nn.init.constant_(regression_model.dense.bias, 0)

    # Load the checkpoints
    reg_model_checkpoint = torch.load(REGRESSION_MODEL_CHECKPOINT_PATH, weights_only=False, map_location=device)
    reg_model_state_dict = reg_model_checkpoint['model_state_dict'] if 'model_state_dict' in reg_model_checkpoint else reg_model_checkpoint
    regression_model.load_state_dict(reg_model_state_dict)
    
    return regression_model

def preprocess_image(image_path):
    # Convert to tensor and normalize
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Convert the image to RGB format
    image = Image.open(image_path).convert('RGB')
    
    # Resize the image
    image = image.resize((640,480))
    
    # Apply the transformations
    image = preprocess(image)
    
    # Add a batch dimension to the tensor
    image = image.unsqueeze(0) 
    
    return image

def predict(image_path):
    # Load the pre-trained model
    model = load()
    
    # Set the model to evaluation mode
    model.eval()
    
    # Preprocess the input image
    input_image = preprocess_image(image_path)
    
    # Move the input image to the selected device
    input_image = input_image.to(device)
    
    # Predict with the model
    with torch.no_grad():
        output = model(input_image)
    
    # Return the predicted number
    return output.item()