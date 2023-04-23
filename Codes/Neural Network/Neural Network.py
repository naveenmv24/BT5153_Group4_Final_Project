    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.preprocessing import LabelEncoder
    import os
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import f1_score
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    
    data = pd.read_csv("C:\\Users\\Ariann\\Downloads\\3593_song_newfeatures_genre_dataset.csv")
    data = data[(data['genre'] != 'Vocal') & (data['genre'] != 'Folk')]
    X = data.drop(["id", "title", "genre", "Unnamed: 0"], axis=1)
    y = data["genre"]
    
    
    desired_samples = y.value_counts()[0]
    
    sampling_strategy = {
        'Blues': desired_samples,
        'Country': desired_samples,
        'Electronic': desired_samples,
     #   'Folk': desired_samples,
        'International': desired_samples,
        'Jazz': desired_samples,
        'Latin': desired_samples,
        'Pop_Rock': desired_samples,
        'Rap': desired_samples,
        'Reggae': desired_samples,
        'RnB': desired_samples,
       # 'Vocal': desired_samples
    }
    
    sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=2023)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    label_encoder = LabelEncoder()
    y_resampled = label_encoder.fit_transform(y_resampled)
    
    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=2023)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    class AudioDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y
            
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    train_dataset = AudioDataset(X_train_tensor, y_train_tensor)
    val_dataset = AudioDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print("Train loader length:", len(train_loader))
    print("Validation loader length:", len(val_loader))
    
    
    class GenreClassifier(nn.Module):
        def __init__(self, input_size, num_classes):
            super(GenreClassifier, self).__init__()
            
            self.fc1 = nn.Linear(input_size, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        
    device = torch.device("cpu")
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    model = GenreClassifier(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    
    num_epochs = 50
    
    print("y_train min:", y_train_tensor.min().item())
    print("y_train max:", y_train_tensor.max().item())
    
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            if batch_y.min() < 0 or batch_y.max() >= num_classes:
                print("Invalid target value(s) detected:")
                print(batch_y)
                continue
            
            outputs = model(batch_X.float())
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()
    
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                outputs = model(batch_X.float())
                _, predicted = torch.max(outputs, 1)
                y_true.extend(batch_y.tolist())
                y_pred.extend(predicted.tolist())
        
        f1 = f1_score(y_true, y_pred, average='weighted')
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, F1 score: {f1:.4f}")
    
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X.float())
            _, predicted = torch.max(outputs, 1)
            y_true.extend(batch_y.tolist())
            y_pred.extend(predicted.tolist())
    
    final_f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Final Validation F1 score: {final_f1:.4f}")