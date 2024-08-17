import torch
import torchvision.models as models
import os

# Directory where the model weights will be saved
save_dir = './cifar10_models/state_dicts/'

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Function to download and save a model
def download_and_save(model_func, model_name):
    save_path = os.path.join(save_dir, f'{model_name}.pt')
    
    if os.path.exists(save_path):
        print(f'{model_name} model already exists. Skipping download.')
    else:
        model = model_func(pretrained=True)
        torch.save(model.state_dict(), save_path)
        print(f'{model_name} model downloaded and saved successfully.')

# List of models to download
models_to_download = {
    'densenet121': models.densenet121,
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'mobilenet_v2': models.mobilenet_v2
}

# Download and save each model
for model_name, model_func in models_to_download.items():
    download_and_save(model_func, model_name)
