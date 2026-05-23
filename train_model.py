"""
ANN 
Glass Identification Model Training Script

This script trains a simple Artificial Neural Network (ANN) on the Glass dataset
from sklearn.datasets and saves the trained model for use in the Streamlit app.

The study of classification of types of glass was motivated by criminological investigation.
At the scene of the crime, the glass left can be used as evidence...if it is correctly identified!

"""

import numpy as np
import pandas as pd
import pickle
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


def load_and_prepare_data():
    """
    Load the Glass dataset and prepare it for training.
    
    Returns:
        X_train, X_test, y_train, y_test: Split and scaled data
        scaler: Fitted StandardScaler for later use
        feature_names: Names of the features
        target_names: Names of the glass classes
    """
    print("🔄 Loading Glass dataset...")
    file_path = r"Dataset/glass.data"

    glass = pd.read_csv(file_path, header=None)

    glass.columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    
    glass.set_index("Id", inplace=True)
        
    X = glass.iloc[:, :-1]

    y = glass.Type
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(glass.Type)
    y = pd.Series(y_encoded, index=glass.index) # Keep it as a pandas Series
    
    # 2. Re-map target names based on the actual unique values found
    glass_type_mapping = {
        1: "building_windows_float_processed",
        2: "building_windows_non_float_processed",
        3: "vehicle_windows_float_processed",
        4: "vehicle_windows_non_float_processed",
        5: "containers",
        6: "tableware",
        7: "headlamps"
    }
    # le.classes_ contains the original labels [1, 2, 3, 5, 6, 7] in sorted order
    target_names = [glass_type_mapping[orig_class] for orig_class in le.classes_]
    
    print(f"✅ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {target_names}")
    print(f"   Features: {list(glass.columns)}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Data split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"✅ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, glass.columns, target_names


class GlassANN(nn.Module):
    """
    Simple ANN model for glass identification.
    
    Architecture:
    - Input Layer: input_dim features
    - Hidden Layer 1: 128 neurons, ReLU activation
    - Dropout: 20% to prevent overfitting
    - Hidden Layer 2: 64 neurons, ReLU activation
    - Dropout: 20% to prevent overfitting
    - Output Layer: num_classes neurons, Softmax activation
    """
    def __init__(self, input_dim, num_classes):
        super(GlassANN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.layer3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.layer3(x)
        return x


def build_ann_model(input_dim, num_classes):
    """
    Build a simple ANN architecture for glass identification.
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    model = GlassANN(input_dim, num_classes)
    
    print("\n🏗️  ANN Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16):
    """
    Train the ANN model with early stopping.
    
    Args:
        model: PyTorch model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        
    Returns:
        Training history
    """
    print(f"\n🚀 Training ANN model (max {epochs} epochs)...")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    
    print(f"Using device: {device}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_acc = 100 * (val_predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_model(model, X_test, y_test, target_names):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features
        y_test: Test labels
        target_names: Names of the target classes
    """
    print("\n📊 Model Evaluation:")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, y_pred = torch.max(outputs.data, 1)
        y_pred = y_pred.cpu().numpy()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return accuracy


def save_model_and_scaler(model, scaler, feature_names, target_names):
    """
    Save the trained model and scaler for use in the Streamlit app.
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        target_names: List of target class names
    """
    print("\n💾 Saving model and artifacts...")
    
    # Save the PyTorch model
    torch.save(model.state_dict(), 'glass_ann_model.pth')
    # Also save the full model architecture info
    model_info = {
        'input_dim': model.layer1.in_features,
        'num_classes': model.layer3.out_features
    }
    with open('model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print("   ✅ Model saved as 'glass_ann_model.pth'")
    print("   ✅ Model info saved as 'model_info.pkl'")
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("   ✅ Scaler saved as 'scaler.pkl'")
    
    # Save feature names and target names
    metadata = {
        'feature_names': feature_names,
        'target_names': target_names
    }
    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    print("   ✅ Metadata saved as 'metadata.pkl'")


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("   ANN GLASS IDENTIFICATION - MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names, target_names = load_and_prepare_data()
    
    # Build the model
    input_dim = X_train.shape[1]
    num_classes = len(target_names)
    model = build_ann_model(input_dim, num_classes)
    
    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=16)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test, target_names)
    
    # Save the model and artifacts
    save_model_and_scaler(model, scaler, feature_names, target_names)
    
    print("\n" + "=" * 60)
    print("   ✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"   Final Test Accuracy: {accuracy:.4f}")
    print("\n   Next steps:")
    print("   1. Run the Streamlit app: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()