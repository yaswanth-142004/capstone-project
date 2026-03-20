import os
import argparse
from datasets import load_from_disk
import torch
import evaluate
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperProcessor,
    WhisperForConditionalGeneration
)
from peft import LoraConfig, get_peft_model

def get_peft_whisper_model(model_name, tokenizer_len):
    """ Loads Whisper, resizes embeddings, and applies LoRA """
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.resize_token_embeddings(tokenizer_len)

    # Typical LoRA config for Whisper
    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def finetune_whisper(dataset_dir, output_dir, model_name="openai/whisper-small"):
    print(f"Loading dataset from {dataset_dir}...")
    dataset = load_from_disk(dataset_dir)
    
    # Simple split
    dataset = dataset.train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    eval_ds = dataset["test"]
    
    print(f"Loading tokenizer and processor for {model_name}...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizerFast.from_pretrained(model_name, language="Telugu", task="transcribe")
    
    # ----------------------------------------------------
    # ADD EXPLICIT BOUNDARY TOKENS FOR CODE-SWITCH TAGGING
    # ----------------------------------------------------
    new_tokens = ["[", "]"]
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"Added {num_added_toks} boundary tokens for Code-Switch tagging.")
    
    processor = WhisperProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # Load model with PEFT
    model = get_peft_whisper_model(model_name, len(tokenizer))
    
    # (Placeholder) Data collator logic to prepare audio/labels
    # real implementation would map dataset to inputs/labels using processor

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=50,
        max_steps=500,
        evaluation_strategy="steps",
        fp16=True, # assumes GPU availability
        predict_with_generate=True,
        generation_max_length=225,
        logging_steps=25,
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    
    print("Setup complete. Model is ready for PEFT trainer.")
    print("In a real run, Seq2SeqTrainer(model, training_args, train_dataset=train_ds, ...).train() would be called here.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Whisper with PEFT to support Tagged Code-Mixed ASR")
    parser.add_argument("--dataset", type=str, required=True, help="Path to HF dataset directory")
    parser.add_argument("--output", type=str, default="models/whisper-te-en-peft")
    parser.add_argument("--model", type=str, default="openai/whisper-small")
    args = parser.parse_args()
    
    finetune_whisper(args.dataset, args.output, args.model)
