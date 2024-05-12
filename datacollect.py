from utils import *
import pickle
from datasets import load_dataset
dataset = load_dataset("roneneldan/TinyStories", data_files="TinyStories-train.txt")
train_data = []
for story in dataset["train"]:
    if len(story["text"].split(" ")) > 200:
        continue
    train_data.append(story["text"])
    if len(train_data) > 100_000:
        break
with open("tinystories.pkl","wb") as f:
    pickle.dump(train_data,f)
print(len(train_data))

