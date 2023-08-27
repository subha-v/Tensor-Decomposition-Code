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


for i in range (0,2):
    loaded_layers, loaded_layer_names = create_and_save_weights_vit(model, f"/content/weights_{i}/")

    # We use i to keep track of the folder number
    # Fully decomposing the model

    reshape_and_save_weights(model, num_layers, loaded_layers, loaded_layer_names, i)

    # Decompose weights until the accuracy goes below 80%
    low_accuracy_model = iterative_compression_with_threshold(model, num_layers, list_of_layers, 0.9, i)

    # Retraining the model to regain accuracy
    model = retrain_vit_model(low_accuracy_model, prepared_ds, 0.2)