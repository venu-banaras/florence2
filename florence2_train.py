# export HF_MODULES_CACHE=/raid/
# export HF_TOKENIZERS_CACHE=/raid/
# export HF_DATASETS_CACHE=/raid/
# export TRANSFORMERS_CACHE=/raid/
# export TORCH_HOME=/raid/

import io
import os
import re
import torch
import html
import base64
import itertools

import numpy as np
# from IPython.core.display import display, HTML
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    AutoModelForCausalLM,
    AutoProcessor,
    get_scheduler
)
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Generator
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from florence2_dataloader import DetectionDataset
import torch
from transformers import AutoModelForCausalLM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_id = 'microsoft/Florence-2-large'
#model_id = 'microsoft/Florence-2-base'

ALL_TRAIN_LOSS = []
# Load model onto the determined device
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(DEVICE)

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Initiate DetectionsDataset and DataLoader for train and validation subsets
BATCH_SIZE = 2
NUM_WORKERS = 0

def collate_fn(batch):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True).to(DEVICE)
    return inputs, answers

# train_dataset = DetectionDataset(
#     jsonl_file_path = "/content/objectdetection-5/train/images/train_annotations.json",
#     image_directory_path = "/content/objectdetection-5/train/images"
# )


train_dataset = DetectionDataset(
    jsonl_file_path = "phsku110k.json",
    image_directory_path = "phsku"
)

# val_dataset = DetectionDataset(
#     jsonl_file_path = "/content/objectdetection-5/valid/images/val_annotations.json",
#     image_directory_path = "/content/objectdetection-5/valid/images"
# )

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)
val_loader = None

# Setup LoRA Florence-2 model

config = LoraConfig(
    r=8,
    lora_alpha=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear", "Conv2d", "lm_head", "fc2"],
    task_type="CAUSAL_LM",
    lora_dropout=0.05,
    bias="none",
    inference_mode=False,
    use_rslora=True,
    init_lora_weights="gaussian",
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()


# training loop

def train_model(train_loader, val_loader, model, processor, epochs=10, lr=1e-6):
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # florence2_inference_results(peft_model, val_loader.dataset, 3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for inputs, answers in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}"):

            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                truncation = True,
                return_token_type_ids=False
            ).input_ids.to(DEVICE)

            outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward(), optimizer.step(), lr_scheduler.step(), optimizer.zero_grad()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss}")
        ALL_TRAIN_LOSS.append(avg_train_loss)

        # model.eval()
        # val_loss = 0
        # with torch.no_grad():
        #     for inputs, answers in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}"):

        #         input_ids = inputs["input_ids"]
        #         pixel_values = inputs["pixel_values"]
        #         labels = processor.tokenizer(
        #             text=answers,
        #             return_tensors="pt",
        #             padding=True,
        #             return_token_type_ids=False
        #         ).input_ids.to(DEVICE)

        #         outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
        #         loss = outputs.loss

        #         val_loss += loss.item()

        #     avg_val_loss = val_loss / len(val_loader)
        #     print(f"Average Validation Loss: {avg_val_loss}")


        output_dir = f"model_checkpoints/ph_ckpt/epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)


#  Training

EPOCHS = 101
LR = 1e-6
train_model(train_loader, val_loader, peft_model, processor, epochs=EPOCHS, lr=LR)
plt.plot(ALL_TRAIN_LOSS, 'r')
plt.savefig("train_loss.jpg")