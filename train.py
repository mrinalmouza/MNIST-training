import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torch.utils.data import DataLoader
import random
import json
from flask import Flask, render_template, jsonify
from model import MnistCNN
import threading
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Flask app
app = Flask(__name__)
training_history = {'loss': [], 'accuracy': []}

# MNIST Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/')
def home():
    return render_template('monitor.html')

@app.route('/get_training_history')
def get_training_history():
    return jsonify(training_history)

def train_model():
    model = MnistCNN().to(device)
    summary(model, (1, 28, 28))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10  # This becomes the maximum number of epochs
    target_accuracy = 98.0  # Stop at 96%
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 100 == 0:
                batch_loss = running_loss / 100
                batch_acc = 100 * correct / total
                training_history['loss'].append(batch_loss)
                training_history['accuracy'].append(batch_acc)
                print(f'Iteration [{i+1}] Loss: {batch_loss:.4f} Accuracy: {batch_acc:.2f}%')
                
                # Check if accuracy threshold is reached
                if batch_acc >= target_accuracy:
                    print(f'\nTarget accuracy of {target_accuracy}% reached!')
                    print(f'Training stopped at epoch {epoch+1}, iteration {i+1}')
                    # Save model
                    torch.save(model.state_dict(), 'mnist_cnn.pth')
                    return  # Exit the training loop
                
                running_loss = 0.0
                correct = 0
                total = 0
    
    # Save model if training completes without reaching target accuracy
    torch.save(model.state_dict(), 'mnist_cnn.pth')
    
    # Generate and save test results
    model.eval()
    test_images = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Get 10 random test images
            for i in range(10):
                idx = random.randint(0, len(images)-1)
                img = images[idx].cpu().numpy().reshape(28, 28)
                pred = predicted[idx].item()
                true = labels[idx].item()
                
                plt.figure(figsize=(3, 3))
                plt.imshow(img, cmap='gray')
                plt.title(f'Pred: {pred}, True: {true}')
                
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode()
                test_images.append(img_str)
                plt.close()
                
    with open('static/test_results.json', 'w') as f:
        json.dump(test_images, f)

def start_server():
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    # Start training in a separate thread
    training_thread = threading.Thread(target=train_model)
    training_thread.start()
    
    # Start Flask server
    start_server() 