from utils.methods import *
from datasets import load_dataset

dataset = load_dataset("cifar10")

data = ["airplane", "automobile", "bird", "cat","deer","dog","frog", "horse", "ship", "truck"]

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