from utils.methods import *
import numpy as np
from utils.weights import *

def reshape_and_save_weights(model, num_layers, loaded_layers, loaded_layer_names):
    for i in range(0, num_layers):
        reshape_weights(model, i, 0.5, loaded_layers, loaded_layer_names)
    
matrix_hats_dict = {}

def iteratively_decompose(model, num_layers):
    for i in range(0, num_layers):
        temp_array = np.load(f"/content/vit_decomposed/layer_{i}_matrix.npy")
        matrix_hats_dict[i] = temp_array

    decomposed_model = update_multiple_layers(model, matrix_hats_dict)
    return decomposed_model


def update_single_layer(model, matrix_hat, layer_num, loaded_layer_names):

  matrix_hat = torch.from_numpy(matrix_hat)
  layer_string = loaded_layer_names[layer_num]


  print("Shape of decomposed weight", matrix_hat.shape)

  # Getting model subset
  layer_component_array = split_string_by_period(layer_string)
  model_subset = get_subset_of_model(layer_component_array, model)

  with torch.no_grad():
    layer = model_subset

  layer.data.copy_(matrix_hat)

  model.save_pretrained(f"/content/decomposed_layer{layer_num}.pt")

  # Reading Model
  decomposed_model = ViTForImageClassification.from_pretrained(
      f"/content/decomposed_layer{layer_num}.pt"
  )

  return decomposed_model

