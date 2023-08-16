from methods import *
import numpy as np
from weights import *

def save_reshaped_weights(model, num_layers):
    for i in range(0, num_layers):
        reshape_weights(model, i, 0.5)
    
matrix_hats_dict = {}

def iteratively_decompose(model, num_layers):
    for i in range(0, num_layers):
        temp_array = np.load(f"/content/vit_decomposed/layer_{i}_matrix.np.npy")
        matrix_hats_dict[i] = temp_array

    decomposed_model = update_multiple_layers(model, matrix_hats_dict)
    return decomposed_model

   