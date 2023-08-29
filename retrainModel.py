from transformers import TrainingArguments
from transformers import Trainer
from utils.dataset import *
import pandas as pd

def freeze_layers(model, n):
    for i, param in enumerate(model.parameters()):
        if i < n:
            param.requires_grad = False
    return model

def retrain_vit_model_with_tracking(model, prepared_ds, epochs=2, output_dir="/content/drive/MyDrive/UCSB 2023/Code/ViT/saved_models"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=epochs,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    accuracies = []

    trainer = Trainer(
        model=model.to(device),
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_ds["train"],
        eval_dataset=prepared_ds["test"],
        tokenizer=feature_extractor,
    )

    for epoch in range(epochs):
        train_results = trainer.train()
        eval_results = trainer.evaluate()

        accuracies.append(eval_results['eval_accuracy'])
        df = pd.DataFrame({'Epoch': list(range(1, epoch + 2)), 'Accuracy': accuracies})
        df.to_csv(output_dir + '/accuracy_tracking.csv', index=False)

    trainer.save_model()
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    best_model_dir = trainer.state.best_model_checkpoint
    updated_model = model.__class__.from_pretrained(best_model_dir)

    return updated_model