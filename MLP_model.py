import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

class FederatedHeartMLP(nn.Module):
    def __init__(self, input_size=13): 
        super(FederatedHeartMLP, self).__init__()
        
        # 1. First Hidden Layer
        self.layer1 = nn.Linear(input_size, 32)
        self.relu1 = nn.ReLU()
        
        # 2. First Regularization: 30% Dropout
        self.dropout1 = nn.Dropout(p=0.3)
        
        # 3. Second Hidden Layer
        self.layer2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        
        # 4. Second Regularization: 20% Dropout
        self.dropout2 = nn.Dropout(p=0.2)
        
        # 5. Output Layer: Single node for binary prediction
        self.output_layer = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Forward pass through layer 1
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # Forward pass through layer 2
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Final output classification (0.0 to 1.0 probability)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        return x
    
def create_MLP_model():
    model = FederatedHeartMLP()
    return model

def train_model(model, dataloader, num_epochs=50):
    """
    Trains on Client Data.

    Returns: Model wieghts
    """

    # Define the Loss Function
    criterion = nn.BCELoss()
    
    # Define the Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- THE TRAINING LOOP ---
    for epoch in range(num_epochs):
        model.train() # Put model in 'training mode'
        
        running_loss = 0.0 
        
        # Iterate over batches in the dataloader
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}")
        
    return model.state_dict()

def test_model(model, dataloader):
    criterion = torch.nn.BCELoss()
    correct, total, loss = 0, 0, 0.0
    
    # Lists to hold all labels and predictions for scikit-learn
    all_labels = []
    all_preds = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            
            # Standardize loss calculation
            loss += criterion(outputs, labels).item() * inputs.size(0)
            
            # Apply a 0.5 threshold for binary classification
            preds = (outputs >= 0.5).float()
            
            # Standard accuracy components
            total += labels.size(0)
            correct += (preds == labels).type(torch.float).sum().item()
            
            # Move tensors to CPU, convert to numpy, and store them
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate overall averages
    avg_loss = loss / total
    accuracy = correct / total
    
    # Calculate advanced metrics
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1
