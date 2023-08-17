from utils.dataset import *
from utils.methods import *
from utils.calculateAccuracy import *
from utils.weights import *
from retrainModel import *
from iterativeCompression import *


model = default_model

# Creating retraining dataset
prepared_ds = create_vit_dataset_for_training()

# Saving and creating weights
create_and_save_weights_vit()

# Fully decomposing the model
reshape_and_save_weights(model, 99)
fully_decomposed_model = iteratively_decompose(model, 99)

# Decompose weights until the accuracy goes below 80%


