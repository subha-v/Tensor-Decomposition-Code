import os
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch

data = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
model_name_or_path = 'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
default_model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=10,  # Assuming CIFAR-10 has 10 classes
        id2label={str(i): c for i, c in enumerate(data)},  # data is the list of classes ['airplane', 'automobile', ...]
        label2id={c: str(i) for i, c in enumerate(data)}
    )

def create_and_save_weights_vit(model=default_model, path="/content/weights/"):
   
    # Load the model
    device = torch.device("cpu")
    model.to(device)

    # Make the path ./content/weights manually
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith(".weight"):
            weight_data = param.detach().numpy()
            # Create the folder weights first
            os.makedirs(path, exist_ok=True)
            file_path = os.path.join(path, f"{name}.npy")
            np.save(file_path, weight_data)
            print(f"Saved weights for {name} to {file_path}")

    model_dir = path  # Specify the directory to save the weight files

    # Read the saved .npy files as individual numpy arrays
    loaded_layers = []
    loaded_layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad and name.endswith(".weight"):
            file_path = os.path.join(model_dir, f"{name}.npy")
            layer_weights = np.load(file_path)
            loaded_layers.append(layer_weights)
            loaded_layer_names.append(name)

    return loaded_layers, loaded_layer_names