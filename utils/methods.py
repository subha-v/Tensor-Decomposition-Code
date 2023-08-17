import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
import sys
import os
import numpy as np
import tensorlearn
import math


def split_string_by_period(string):
    return string.split(".")


def get_subset_of_model(attributes_array, model):
    last_model = model
    for attribute in attributes_array:
        last_model = getattr(last_model, attribute)
    return last_model


def update_multiple_layers(model, matrix_hats_dict, loaded_layer_names):
    for layer_num, matrix_hat in matrix_hats_dict.items():
        if layer_num > 1000:
            continue
        # layer_num = layer_num+101
        # print(matrix_hats_dict)
        print("Updating this layer", layer_num)
        print("Dimensions", matrix_hat.shape)
        # Setting matrices
        matrix_hat = torch.from_numpy(matrix_hat)
        layer_string = loaded_layer_names[layer_num]

        # Getting model subset
        layer_component_array = split_string_by_period(layer_string)
        model_subset = get_subset_of_model(layer_component_array, model)

        # Update Model Weights
        with torch.no_grad():
            layer = (
                model_subset  # Assuming the weights are stored in a "weight" attribute
            )
            layer.copy_(matrix_hat)

        # Saving Model
        model.save_pretrained("/content/decomposed.pt")

        # Reading Model
        decomposed_model = ViTForImageClassification.from_pretrained(
            "/content/decomposed.pt"
        )

        # Update the main model with the decomposed model for the next iteration
        model = decomposed_model

    return decomposed_model


def reshape_weights(model, layer_number, epsilon, loaded_layers, loaded_layer_names):
    original_dimensions = original_dimensions = list(loaded_layers[layer_number].shape)
    layer_component_array = split_string_by_period(loaded_layer_names[layer_number])
    model_subset = get_subset_of_model(layer_component_array, model)
    layer_1_weights_3d = (
        model_subset.detach().cpu().numpy()
    )  # Move tensor to CPU memory
    # Calculate the volume of the array
    volume = np.prod(layer_1_weights_3d.shape)
    layer_string = loaded_layer_names[layer_number]

    dimensions = find_smallest_sum_triplet(volume)

    layer_1_weights_3d = np.reshape(layer_1_weights_3d, dimensions)

    # Perform tensor operations
    factors2 = tensorlearn.auto_rank_tt(layer_1_weights_3d, epsilon)
    layer_1_weights_3d_prime = tensorlearn.tt_to_tensor(factors2)

    # Reshape back to the original size
    matrix_hat = np.reshape(layer_1_weights_3d_prime, original_dimensions)
    print("Reshaped to original", matrix_hat.shape)

    # Saving matrix hat as a numpy array
    dimensions_string = "_".join(map(str, dimensions))

    # Finding Compression Ratio and Space Saving
    compression_ratio = tensorlearn.tt_compression_ratio(factors2)
    space_saving = 1 - (1 / tensorlearn.tt_compression_ratio(factors2))
    print("Compression Ratio:", compression_ratio)
    print("Amount of the original:", space_saving)
    # space_savings.append(space_saving)
    
    os.makedirs("/content/vit_decomposed/", exist_ok=True)
    np.save(f"/content/vit_decomposed/layer_{layer_number}_matrix.np", matrix_hat)

    return epsilon, dimensions, matrix_hat


def find_factors(volume):
    # Find factors of the given volume
    factors = []
    for i in range(1, int(np.sqrt(volume)) + 1):
        if volume % i == 0:
            factors.append(i)
            factors.append(volume // i)
    return factors


def find_dimensions(factors, volume):
    # Find three dimensions that multiply to make the volume
    dimensions = []
    for i in range(len(factors) - 2):
        for j in range(i + 1, len(factors) - 1):
            for k in range(j + 1, len(factors)):
                if (
                    factors[i] * factors[j] * factors[k] == volume
                    and factors[i] >= 2
                    and factors[j] >= 2
                    and factors[k] >= 2
                ):
                    dimensions = [factors[i], factors[j], factors[k]]
                    return dimensions

    print(dimensions)
    return None


def find_smallest_sum_triplet(target_product):
    smallest_sum = sys.maxsize
    result_triplet = None

    for a in range(1, int(target_product ** (1 / 3)) + 1):
        if target_product % a == 0:
            for b in range(a, int((target_product // a) ** 0.5) + 1):
                if (target_product // a) % b == 0:
                    c = target_product // (a * b)

                    # Calculate the sum of the triplet
                    triplet_sum = a + b + c

                    # Update the smallest sum and the result triplet if necessary
                    if triplet_sum < smallest_sum:
                        smallest_sum = triplet_sum
                        result_triplet = [a, b, c]

    return result_triplet


def create_matrix_hats_dict(list_of_layers):
    matrix_hats_dict = {}
    i = 0
    for layer in list_of_layers:
        if list_of_layers[i] > 1000:
            i += 1
            continue
        # print(layer_num)
        temp_array = np.load(
            f"/content/vit_decomposed/layer_{list_of_layers[i]}_matrix.np.npy"
        )
        matrix_hats_dict[i] = temp_array
        i += 1

    # print(matrix_hats_dict)
    return matrix_hats_dict
