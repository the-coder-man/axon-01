import json
import string
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import tkinter as tk
from torch.utils.data import Dataset, DataLoader
import pyttsx3
import torch.nn as nn
class NeuralNet(nn.Module): # loads the chatbot.pth file
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
# Simple tokenizer: split on spaces and strip punctuation
def tokenize(sentence):
    translator = str.maketrans('', '', string.punctuation)
    return sentence.translate(translator).lower().split()

# Basic stemmer (identity function here, can be extended)
def stem(word):
    return word.lower()

def bag_of_words(tokenized_sentence, words):
    sentence_words = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1.0
    return bag

# Load training data from a single JSON file
with open('intents/chatbot_training_data.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!', ',']
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.x_data = np.array(X_train, dtype=np.float32)
        self.y_data = np.array(y_train, dtype=np.int64)
        self.n_samples = len(self.x_data)

    def __getitem__(self, index):
        index = int(index)  # Ensure index is int
        return torch.from_numpy(self.x_data[index]), torch.tensor(self.y_data[index])


    def __len__(self):
        return self.n_samples

batch_size = 8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training started...")
for epoch in range(num_epochs):
    def __getitem__(self, index):
        return torch.from_numpy(self.x_data[index]), torch.tensor(self.y_data[index], dtype=torch.long)
    for (words, labels) in train_loader:
        words = words.to(device).float()  # Ensure inputs are float
        labels = labels.to(device)        # Move labels to the device
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device).float()  # Ensure inputs are float
        labels = labels.to(device)        # Move labels to the device

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    
# Save model data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "chatbot.pth"    #saves the chatbot.pth file
torch.save(data, FILE)

bot_name = "Axon"
try:
    engine = pyttsx3.init()
except Exception as e:
    print("TTS engine failed to start:", e)
    engine = None

def speak(text):
    if engine:
        engine.say(text)
        engine.runAndWait()



def get_response(sentence):
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device).float()  # ensure it's float for model

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    index = int(predicted.item())  # ✅ force to int
    tag = tags[index]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][index]  # ✅ use int index

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."


class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Axon 01")

        self.chat_log = tk.Text(root, bd=1, bg="white", font=("Arial", 12))
        self.chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.entry_box = tk.Entry(root, bd=1, font=("Arial", 12))
        self.entry_box.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.entry_box.bind("<Return>", self.send_message)


    def send_message(self, event):
        user_input = self.entry_box.get()
        if not user_input.strip():
            return
        self.chat_log.insert(tk.END, f"You: {user_input}\n")
        response = get_response(user_input)
        self.chat_log.insert(tk.END, f"{bot_name}: {response}\n")
        self.chat_log.see(tk.END)
        speak(response)
        self.entry_box.delete(0, tk.END)

if __name__ == '__main__':
    root = tk.Tk()
    gui = ChatbotGUI(root)
    root.mainloop()
# axon 01.py
# This code implements a simple chatbot using PyTorch and Tkinter for the GUI.
# It includes a neural network model trained on intents from a JSON file,
# and uses text-to-speech for responses. The GUI allows users to interact with the chatbot
