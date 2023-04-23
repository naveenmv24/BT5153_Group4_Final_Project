import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

data = pd.read_csv("C:\\Users\\Ariann\\Downloads\\3589_song_Rhythm_Histogram_dataset.csv")
data = data[(data['genre'] != 'Vocal') & (data['genre'] != 'Folk')]

X = data.iloc[:, :60].values
y = data["genre"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

desired_samples = 1526  

sampling_strategy = {
    i: desired_samples
    for i in range(len(np.unique(y)))
}

sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=2023)

X_temp, X_val, y_temp, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_temp_2D = X_temp.reshape(X_temp.shape[0], X_temp.shape[1])

X_train_2D, y_train = sampler.fit_resample(X_temp_2D, y_temp)

X_train = X_train_2D.reshape(X_train_2D.shape[0], X_train_2D.shape[1], 1)

X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AudioDataset(X_train_tensor, y_train_tensor)
val_dataset = AudioDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GenreClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
hidden_size = 64
num_layers = 2


num_classes = len(label_encoder.classes_)
model = GenreClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_f1 = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        running_f1 += f1 * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = running_f1 / len(dataloader.dataset)

    return epoch_loss, epoch_f1


def validate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_f1 = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            f1 = f1_score(labels.data.cpu().numpy(), preds.cpu().numpy(), average='weighted')
            running_f1 += f1 * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_f1 = running_f1 / len(dataloader.dataset)

    return epoch_loss, epoch_f1


num_epochs = 20

for epoch in range(num_epochs):
    train_loss, train_f1 = train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1 = validate_model(model, val_loader, criterion, device)

    print(f'Epoch {epoch+1}/{num_epochs}:')
    print(f'Train Loss: {train_loss:.4f} F1: {train_f1:.4f}')
    print(f'Val Loss: {val_loss:.4f} F1: {val_f1:.4f}\n')

