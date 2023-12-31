from utils.methods import *
import numpy as np
from utils.weights import *
from utils.calculateAccuracy import *
import pandas as pd

accuracies_df = pd.DataFrame(columns=['Iteration', 'Decomposed_Accuracy'])

def reshape_and_save_weights(model, num_layers, loaded_layers, loaded_layer_names, folder_num):
    for i in range(0, num_layers):
        reshape_weights(model, i, 0.5, loaded_layers, loaded_layer_names, folder_num)
    
matrix_hats_dict = {}

def iteratively_decompose(model, num_layers, folder_number, loaded_layer_names):
    for i in range(0, num_layers):
        temp_array = np.load(f"/content/vit_decomposed_{folder_number}/layer_{i}_matrix.npy")
        matrix_hats_dict[i] = temp_array

    decomposed_model = update_multiple_layers(model, matrix_hats_dict, loaded_layer_names)
    return decomposed_model


def update_single_layer(model, matrix_hat, layer_num, loaded_layer_names):

  matrix_hat = torch.from_numpy(matrix_hat)
  layer_string = loaded_layer_names[layer_num]


  # print("Shape of decomposed weight", matrix_hat.shape)

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


def iterative_compression_with_threshold(model, num_layers, list_of_layers, accuracy_threshold, loaded_layer_names, folder_num):
  list_of_layers = []
  
  for i in range(0, num_layers):
      list_of_layers.append(i)
      print(list_of_layers)

      matrix_hat = np.load(f"/content/vit_decomposed_{folder_num}/layer_{list_of_layers[i]}_matrix.npy")

      decomposed_model = update_single_layer(model, matrix_hat, i, loaded_layer_names)

      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      decomposed_model.to(device)

      decomposed_accuracy = calculate_accuracy(30, decomposed_model)

      accuracies_df = accuracies_df.append({'Iteration': i, 'Decomposed_Accuracy': decomposed_accuracy}, ignore_index=True)
      accuracies_df.to_excel("/content/accuracies.xls", index=False)  # Save to Excel file

      print("Original List of Layers", list_of_layers)
      print("Decomposed Accuracy", decomposed_accuracy)

      if decomposed_accuracy < accuracy_threshold:
          print(f"Decomposed accuracy is below {accuracy_threshold} for layer {i}. Stopping the loop.")
          return decomposed_model, list_of_layers
          
      else:
          pass
      
      print("Updated List of Layers: ", list_of_layers)

      model = decomposed_model