import torch
import torch.optim as optim
from model import SimpleCNN
from data_loader import load_data
import time  # Import time module

def train(train_dir='dataset/image', val_dir='dataset/validation', model_path='model.pth', num_epochs=100, learning_rate=0.001):
    train_loader = load_data(train_dir, batch_size=32, augment=True)
    val_loader = load_data(val_dir, batch_size=32)

    model_obj = SimpleCNN(num_classes=10)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_obj.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(num_epochs):
        model_obj.train()
        train_loss = 0
        start_time = time.time()  # Start timing

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_obj(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_duration = time.time() - start_time  # End timing

        model_obj.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model_obj(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Duration: {epoch_duration:.2f} seconds')
    
    torch.save(model_obj.state_dict(), model_path)

if __name__ == "__main__":
    train()
