#!/usr/bin/env python
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import json
import time
import random
from scipy.spatial.distance import cosine, euclidean
from sklearn.metrics.pairwise import cosine_similarity

# Available models for fine-tuning
AVAILABLE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "intfloat": "intfloat/multilingual-e5-large",
    "baai": "BAAI/bge-large-en-v1.5",
    "e5": "intfloat/e5-large-v2"
}

# Available similarity metrics
SIMILARITY_METRICS = {
    "cosine": lambda x, y: 1 - cosine(x, y),  # Cosine similarity (default)
    "euclidean": lambda x, y: 1 / (1 + euclidean(x, y)),  # Normalized Euclidean distance
    "dot": lambda x, y: np.dot(x, y),  # Dot product
    "cosine_sklearn": lambda x, y: cosine_similarity([x], [y])[0][0]  # sklearn's cosine similarity
}

class FieldNameDataset(Dataset):
    def __init__(self, contexts, field_names, neg_samples=5):
        self.examples = []
        unique_fields = list(set(field_names))
        
        print("Creating training examples...")
        for context, field_name in tqdm(zip(contexts, field_names), total=len(contexts)):
            # Ensure strings are properly encoded
            context = str(context)
            field_name = str(field_name)
            
            # Add positive example
            self.examples.append(InputExample(texts=[context, field_name], label=1.0))
            
            # Add limited negative examples
            neg_fields = random.sample([f for f in unique_fields if f != field_name], 
                                     min(neg_samples, len(unique_fields)-1))
            for neg_field in neg_fields:
                neg_field = str(neg_field)  # Ensure string
                self.examples.append(InputExample(texts=[context, neg_field], label=0.0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def load_training_data(catalog_file="widget_catalog.csv"):
    if not os.path.exists(catalog_file):
        print(f"Error: Catalog file {catalog_file} not found!")
        return [], []
    
    # Read the catalog file
    df = pd.read_csv(catalog_file)
    
    # Use the correct column names from the catalog and ensure string type
    contexts = df['context'].astype(str).tolist()
    field_names = df['widgetName'].astype(str).tolist()
    
    print(f"Loaded {len(contexts)} examples from catalog")
    print(f"Unique field names: {len(set(field_names))}")
    
    return contexts, field_names

def finetune_model(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                  catalog_file="widget_catalog.csv",
                  output_dir="finetuned_model",
                  batch_size=16,
                  epochs=3,
                  learning_rate=2e-5,
                  neg_samples=5,
                  similarity_metric="cosine"):
    
    start_time = time.time()
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU for training")
    
    # Load the base model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # Load training data
    contexts, field_names = load_training_data(catalog_file)
    
    if not contexts:
        print("No training data found! Make sure the catalog file exists and is properly formatted.")
        return
    
    # Split data into train and validation sets
    train_contexts, val_contexts, train_fields, val_fields = train_test_split(
        contexts, field_names, test_size=0.1, random_state=42
    )
    
    # Create datasets with limited negative samples
    train_dataset = FieldNameDataset(train_contexts, train_fields, neg_samples=neg_samples)
    val_dataset = FieldNameDataset(val_contexts, val_fields, neg_samples=neg_samples)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Create data loaders with memory pinning
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=True if device == "cuda" else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        pin_memory=True if device == "cuda" else False
    )
    
    # Create evaluator
    # Prepare validation data for evaluator
    val_sentences1 = []
    val_sentences2 = []
    val_scores = []
    
    for example in val_dataset:
        val_sentences1.append(str(example.texts[0]))  # context
        val_sentences2.append(str(example.texts[1]))  # field name
        val_scores.append(float(example.label))
    
    evaluator = EmbeddingSimilarityEvaluator(val_sentences1, val_sentences2, val_scores)
    
    # Define loss function
    train_loss = losses.ContrastiveLoss(model)
    
    # Estimate training time BEFORE starting
    examples_per_second = 100 if device == "cuda" else 10  # rough estimate
    total_examples = len(train_dataset) * epochs
    estimated_seconds = total_examples / examples_per_second
    hours = int(estimated_seconds // 3600)
    minutes = int((estimated_seconds % 3600) // 60)
    
    print(f"\nEstimated training time: {hours} hours and {minutes} minutes")
    print("This is a rough estimate and actual time may vary based on your hardware and data")
    
    # Fine-tune the model
    print("\nStarting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=100,
        optimizer_params={'lr': learning_rate},
        show_progress_bar=True
    )
    
    # Save the fine-tuned model
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir)
    
    end_time = time.time()
    total_time = end_time - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\nTraining completed in {hours}h {minutes}m {seconds}s")
    print(f"Model saved to {output_dir}")
    
    # Save training configuration
    config = {
        "base_model": model_name,
        "training_examples": len(contexts),
        "unique_field_names": len(set(field_names)),
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "negative_samples": neg_samples,
        "catalog_file": catalog_file,
        "device_used": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "training_time_seconds": total_time,
        "training_time_human": f"{hours}h {minutes}m {seconds}s",
        "similarity_metric": similarity_metric
    }
    
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minilm", choices=list(AVAILABLE_MODELS.keys()),
                    help="Model to fine-tune (minilm, mpnet, intfloat, baai, e5)")
    ap.add_argument("--catalog", default="widget_catalog.csv",
                    help="Path to the widgets catalog CSV file")
    ap.add_argument("--output_dir", default="finetuned_model",
                    help="Directory to save the fine-tuned model")
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Training batch size")
    ap.add_argument("--epochs", type=int, default=3,
                    help="Number of training epochs")
    ap.add_argument("--learning_rate", type=float, default=2e-5,
                    help="Learning rate for training")
    ap.add_argument("--neg_samples", type=int, default=5,
                    help="Number of negative samples per positive example")
    ap.add_argument("--similarity", default="cosine", choices=list(SIMILARITY_METRICS.keys()),
                    help="Similarity metric to use (cosine, euclidean, dot, cosine_sklearn)")
    args = ap.parse_args()
    
    # Get the full model name from the shorthand
    model_name = AVAILABLE_MODELS[args.model]
    
    # Create output directory based on model name
    output_dir = f"finetuned_{args.model}"
    
    finetune_model(
        model_name=model_name,
        catalog_file=args.catalog,
        output_dir=output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        neg_samples=args.neg_samples,
        similarity_metric=args.similarity
    ) 