#!/usr/bin/env python
# coding=utf-8

"""
@author: zgwi
@date: 2024/8/13 14:29
@source from:

单机单卡来对bert进行情感分类微调任务；

"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

data = pd.read_csv("./ChnSentiCorp_htl_all.csv")

data = data.dropna()


from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
for i in range(5):
    print(dataset[i])

from torch.utils.data import random_split

trainset, validset = random_split(dataset, lengths=[0.9, 0.1])


for i in range(10):
    print(trainset[i])

import torch

tokenizer = AutoTokenizer.from_pretrained("./model")


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels)
    return inputs


from torch.utils.data import DataLoader

trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)


from torch.optim import Adam

model = AutoModelForSequenceClassification.from_pretrained("./model")

if torch.cuda.is_available():
    model = model.cuda()

optimizer = Adam(model.parameters(), lr=2e-5)

import time


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred == batch["labels"]).sum().item()
    acc = acc_num / len(validset)
    return acc


def train(epoch):
    model.train()
    for i in range(epoch):
        start = time.time()
        for step, batch in enumerate(trainloader):
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                print(f"Epoch {i + 1}, Step {step}, Loss: {loss.item()}")
        acc = evaluate()
        end = time.time()
        print(f"Epoch {i + 1} completed. Validation Accuracy: {acc}, Time: {end - start} seconds")

train(3)
