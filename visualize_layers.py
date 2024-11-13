import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from model import MnistCNN
from torchvision import datasets, transforms
from PIL import Image
import seaborn as sns

class LayerVisualizer:
    def __init__(self, model_path='mnist_cnn.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MnistCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_sample_digits(self):
        """Get one sample of each digit from MNIST test set"""
        test_dataset = datasets.MNIST('./data', train=False, 
                                    transform=transforms.ToTensor())
        samples = {}
        for img, label in test_dataset:
            if label not in samples:
                samples[label] = img
            if len(samples) == 10:
                break
        return samples

    def visualize_first_layer_filters(self):
        """Visualize filters of the first convolutional layer"""
        # Get the first conv layer weights
        weights = self.model.conv_layers[0].weight.data.cpu()
        n_filters = weights.shape[0]  # Get actual number of filters
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(n_filters)))
        
        # Plot all filters
        plt.figure(figsize=(10, 10))
        for i in range(n_filters):
            plt.subplot(grid_size, grid_size, i+1)
            plt.imshow(weights[i, 0], cmap='viridis')
            plt.axis('off')
        plt.suptitle(f'First Layer Filters (3x3 kernels)\nTotal filters: {n_filters}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        # Print shapes for verification
        print(f"\nFirst layer filter shapes:")
        print(f"Number of filters: {weights.shape[0]}")
        print(f"Input channels: {weights.shape[1]}")
        print(f"Filter size: {weights.shape[2]}x{weights.shape[3]}")

    def visualize_feature_maps(self, digit, digit_label):
        """Visualize feature maps for each layer for a given digit"""
        # Prepare input
        digit = digit.unsqueeze(0).to(self.device)
        
        # Get activations for each conv layer
        activations = []
        x = digit
        
        # Store intermediate activations
        for i in range(4):  # 4 conv blocks
            # Conv + ReLU
            x = self.model.conv_layers[i*3](x)  # Conv2d
            x = self.model.conv_layers[i*3 + 1](x)  # ReLU
            activations.append(x.detach().cpu())
            # MaxPool
            x = self.model.conv_layers[i*3 + 2](x)
        
        # Plot original digit and feature maps
        plt.figure(figsize=(20, 4))
        
        # Original digit
        plt.subplot(1, 5, 1)
        plt.imshow(digit.squeeze().cpu(), cmap='gray')
        plt.title(f'Original Digit: {digit_label}')
        plt.axis('off')
        
        # Feature maps for each layer
        for i, activation in enumerate(activations):
            plt.subplot(1, 5, i+2)
            # Take mean across channels for visualization
            mean_activation = activation.squeeze().mean(dim=0)
            plt.imshow(mean_activation, cmap='viridis')
            plt.title(f'Layer {i+1} Features\n({activation.shape[1]} channels)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

        # Optional: Show detailed channel activations for each layer
        for i, activation in enumerate(activations):
            n_channels = activation.shape[1]
            if n_channels > 16:  # If too many channels, show only first 16
                n_channels = 16
            
            plt.figure(figsize=(12, 8))
            plt.suptitle(f'Layer {i+1} Individual Channel Activations', fontsize=14)
            
            for j in range(n_channels):
                plt.subplot(4, 4, j+1)
                channel_activation = activation.squeeze()[j]
                plt.imshow(channel_activation, cmap='viridis')
                plt.title(f'Channel {j+1}')
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()

    def visualize_all_digits(self):
        """Visualize feature maps for all digits"""
        samples = self.get_sample_digits()
        
        for digit_label, img in samples.items():
            print(f"\nProcessing digit {digit_label}:")
            self.visualize_feature_maps(img, digit_label)
            
            # Optional: Add a pause between digits
            input("Press Enter to continue to next digit...")

    def visualize_layer_responses(self):
        """Visualize how each layer responds to different digits"""
        # Get one sample of each digit
        samples = self.get_sample_digits()
        
        # Get activations for each digit
        layer_responses = {i: [] for i in range(4)}  # 4 layers
        
        for digit, img in samples.items():
            x = img.unsqueeze(0).to(self.device)
            
            # Get responses from each layer
            for i in range(4):
                # Apply conv block
                x = self.model.conv_layers[i*3](x)  # Conv2d
                x = self.model.conv_layers[i*3 + 1](x)  # ReLU
                # Store mean activation
                mean_response = x.detach().cpu().mean().item()
                layer_responses[i].append(mean_response)
                # Apply MaxPool
                x = self.model.conv_layers[i*3 + 2](x)
        
        # Plot responses
        plt.figure(figsize=(15, 5))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.bar(range(10), layer_responses[i])
            plt.title(f'Layer {i+1} Response')
            plt.xlabel('Digit')
            plt.ylabel('Mean Activation')
        plt.tight_layout()
        plt.show()

    def visualize_decision_making(self):
        """Visualize the decision-making process for each digit"""
        samples = self.get_sample_digits()
        
        for digit, img in samples.items():
            print(f"\nAnalyzing digit {digit}:")
            
            x = img.unsqueeze(0).to(self.device)
            activations = []
            
            # Forward pass with hooks
            def hook_fn(module, input, output):
                activations.append(output.detach().cpu())
            
            hooks = []
            # Register hooks for conv layers
            for i in range(4):
                hook = self.model.conv_layers[i*3].register_forward_hook(hook_fn)
                hooks.append(hook)
            
            # Forward pass
            with torch.no_grad():
                output = self.model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # Plot
            plt.figure(figsize=(15, 5))
            
            # Original image
            plt.subplot(1, 5, 1)
            plt.imshow(img.squeeze(), cmap='gray')
            plt.title(f'Input: {digit}')
            plt.axis('off')
            
            # Feature maps
            for i, activation in enumerate(activations):
                plt.subplot(1, 5, i+2)
                mean_activation = activation.squeeze().mean(dim=0)
                plt.imshow(mean_activation, cmap='viridis')
                plt.title(f'Layer {i+1}\nResponse')
                plt.axis('off')
            
            # Probabilities
            plt.subplot(1, 5, 5)
            probs = probabilities.squeeze().cpu().numpy()
            plt.bar(range(10), probs)
            plt.title('Output\nProbabilities')
            plt.xlabel('Digit')
            
            plt.tight_layout()
            plt.show()
            
            # Remove hooks
            for hook in hooks:
                hook.remove()

# Usage
if __name__ == "__main__":
    visualizer = LayerVisualizer('mnist_cnn.pth')
    
    print("\n1. Visualizing First Layer Filters")
    visualizer.visualize_first_layer_filters()
    
    print("\n2. Visualizing Feature Maps for each digit")
    visualizer.visualize_all_digits()
    
    print("\n3. Visualizing Layer Responses across digits")
    visualizer.visualize_layer_responses()
    
    print("\n4. Visualizing Decision Making Process")
    visualizer.visualize_decision_making() 