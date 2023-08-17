from utils.dataset import *
from utils.methods import *
from utils.calculateAccuracy import *
from utils.weights import *
from retrainModel import *
from iterativeCompression import *

model = default_model
num_layers = 99
accuracies = []
list_of_layers = []
j=0

# Creating retraining dataset
prepared_ds = create_vit_dataset_for_training()

# Saving and creating weights
create_and_save_weights_vit()

# Fully decomposing the model
reshape_and_save_weights(model, num_layers)
fully_decomposed_model = iteratively_decompose(model, num_layers)

# Decompose weights until the accuracy goes below 80%
list_of_layers = []

for i in range(0, num_layers):
    list_of_layers.append(i)
    print(list_of_layers)

    matrix_hat = np.load(f"/content/vit_decomposed/layer_{list_of_layers[i]}_matrix.np.npy")

    decomposed_model = update_single_layer(model, matrix_hat, i)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    decomposed_model.to(device)

    decomposed_accuracy = calculate_accuracy(30, decomposed_model)

    print("Original List of Layers", list_of_layers)
    print("Decomposed Accuracy", decomposed_accuracy)

    if decomposed_accuracy < 0.8:
        print(f"Decomposed accuracy is below 0.8 for layer {i}. Stopping the loop.")
    else:
        pass

    j+=1

    print("Updated List of Layers: ", list_of_layers)

    model = decomposed_model