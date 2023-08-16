from methods import *
import os
import numpy as np
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset
from transformers import ViTFeatureExtractor
import torch
import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")
model_name_or_path = 'aaraki/vit-base-patch16-224-in21k-finetuned-cifar10'
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name_or_path)

def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

def process_example(example):
    inputs = feature_extractor(example['img'], return_tensors='pt')
    inputs['label'] = example['label']
    return inputs

def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch['img']], return_tensors='pt')

    # Don't forget to include the labels!
    inputs['label'] = example_batch['label']
    return inputs

def collate_fn(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['label'] for x in batch])
    }

def create_vit_dataset_for_training():
    dataset = load_dataset("cifar10")
    data = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    data_labels = []
    data_idx = []

    for i in range(10000):
        data_idx.append(dataset['train'][i]['label'])
        data_labels.append(data[dataset['train'][i]['label']])

    labels = dataset['train'].features['label']
    prepared_ds = dataset.with_transform(transform)

    return prepared_ds
    


