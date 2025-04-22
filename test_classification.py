import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from keras.preprocessing.sequence import pad_sequences

# Load dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
raw_data = data_dict['data']  # Raw input data
labels = np.asarray(data_dict['labels'])

# Handle inconsistent lengths by padding the sequences
max_len = 42  # Max length of landmarks (21 landmarks * 2 coordinates)
data_padded = pad_sequences(raw_data, maxlen=max_len, padding='post', dtype='float32')

# Convert to NumPy array
data = np.asarray(data_padded, dtype=np.float32)

# Encode string labels into integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(data_tensor, labels_tensor, test_size=0.2, stratify=labels)

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CNNClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(CNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# SAC Model (Policy Learning)
class SACModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SACModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action


num_features = data.shape[1]
num_classes = len(np.unique(labels))
model = CNNClassifier(num_features, num_classes)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the CNN Model
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate Model
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.numpy())
        y_true.extend(targets.numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the Trained Model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, f)
