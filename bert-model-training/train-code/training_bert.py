import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    AutoModelForSequenceClassification, 
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Environment Setup
MODEL_ROOT = Path("model/finetuned")
MODEL_ROOT.mkdir(exist_ok=True, parents=True)
print("Model Output Directory:", MODEL_ROOT)

# Configuration parameters

# BASE_DIR = "/Users/ram/Desktop/AI-Team-Work/ram-work/bert-training/train-code"
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
# MODEL_PATH = os.path.join(BASE_DIR, "model/finetuned/checkpoint-435")
MAX_LENGTH = 512
BASE_DIR = "/Users/ram/Desktop/AI-Team-Work/ram-work/bert-training/train-code"
MODEL_PATH = os.path.join(BASE_DIR, "model/finetuned/checkpoint-435")

def seed_everything(seed: int):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def compute_metrics(p):
    preds, labels = p
    preds = preds.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(laAI-Team-Work/ram-work/bert-trainingbels, preds, average='weighted'),
        'f1': f1_score(labels, preds, average='weighted'),
    }

def main():
    # Set random seed
    seed_everything(42)

    # Check if MPS is available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    df_train = pd.read_json('test_data_with_id.json')
    print("Total samples:", len(df_train))

    # Convert category_id to numerical labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # df_train['labels'] = le.fit_transform(df_train['category_id'])
    df_train['labels'] = df_train['category_id']
    print(df_train['labels'])
    print(df_train['labels'].unique())
    print(df_train['labels'].value_counts())
    print(df_train.head())

    # Split data
    train_df, test_df = train_test_split(df_train, test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def tokenize(sample):
        return tokenizer(sample['sentence'], max_length=MAX_LENGTH, truncation=True)

    # Prepare datasets
    ds_train = Dataset.from_pandas(train_df)
    ds_eval = Dataset.from_pandas(test_df)

    ds_train = ds_train.map(tokenize).remove_columns(['sentence','category_id' ,'__index_level_0__'])
    ds_eval = ds_eval.map(tokenize).remove_columns(['sentence','category_id' ,'__index_level_0__'])

    # Training arguments optimized for MPS
    training_args = TrainingArguments(
        output_dir=str(MODEL_ROOT),
        overwrite_output_dir=True,
        learning_rate=1e-5,
        num_train_epochs=15,
        do_eval=True,
        eval_steps=25,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        report_to="none",
        save_total_limit=1,
        save_strategy="steps",
        save_steps=25,
        logging_steps=25,
        lr_scheduler_type='linear',
        metric_for_best_model="accuracy",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
        save_safetensors=True,
        # Disable fp16 for MPS
        fp16=False,
        # Disable pin_memory for MPS
        dataloader_pin_memory=not torch.backends.mps.is_available()
    )

    # Load model
    num_labels = len(df_train['labels'].unique())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=num_labels)
    model = model.to(device)

    # Initialize trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # Keep tokenizer for now as processing_class is not fully supported yet
        tokenizer=tokenizer
    )

    # Start training
    print("Starting training...")
    trainer.train()

    # Evaluate model
    print("\nEvaluating model...")
    metrics = trainer.evaluate()
    print("\nEvaluation metrics:")
    print(json.dumps(metrics, indent=2))

    # Save final model and tokenizer
    final_output_dir = MODEL_ROOT / "final_model"
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"\nModel and tokenizer saved to {final_output_dir}")

if __name__ == "__main__":
    main() 