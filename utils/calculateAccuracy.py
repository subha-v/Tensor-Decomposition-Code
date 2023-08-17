from utils.methods import *
from datasets import load_dataset

dataset = load_dataset("cifar10")

data = ["airplane", "automobile", "bird", "cat","deer","dog","frog", "horse", "ship", "truck"]
feature_extractor = ViTFeatureExtractor.from_pretrained('aaraki/vit-base-patch16-224-in21k-finetuned-cifar10')
data_labels = []
data_idx = []

for i in range (10000):
  data_idx.append(dataset['test'][i]['label'])
  data_labels.append(data[dataset['test'][i]['label']])


def is_prediction_correct(max_prob_index, i):
  if (data_idx[max_prob_index.item()]==dataset['test'][i]['label']):
    return True
  else:
    return False

def calculate_accuracy(total_examples, model):
    correct_predictions = 0

    for i in range(total_examples):
        image = dataset['test'][i]['img']  # Assuming the dataset contains 'img' and 'label' keys
        label = dataset['test'][i]['label']

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors='pt')

        # Move the input tensor to GPU (if available)
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}

        # Run inference
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits).item()

        # Compare predicted and ground truth labels
        if predicted_label == label:
            correct_predictions += 1

    accuracy = correct_predictions / total_examples
    return accuracy