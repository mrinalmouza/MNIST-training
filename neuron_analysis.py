import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from collections import defaultdict
from model import MnistCNN

class NeuronAnalyzer:
    def __init__(self, model_path='mnist_cnn.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MnistCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load test dataset
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
            batch_size=100, shuffle=False
        )

    def analyze_neuron_activations(self, layer_idx, num_samples=1000):
        """Analyze neuron activations for specific layer"""
        activations = []
        labels = []
        
        # Collect activations
        with torch.no_grad():
            for images, batch_labels in self.test_loader:
                if len(labels) >= num_samples:
                    break
                    
                images = images.to(self.device)
                x = images
                
                # Forward pass until target layer
                for i in range(layer_idx * 3 + 2):
                    x = self.model.conv_layers[i](x)
                
                # Store activations and labels
                activations.append(x.cpu())
                labels.extend(batch_labels.numpy())
        
        # Concatenate all activations
        activations = torch.cat(activations, 0)
        labels = np.array(labels[:num_samples])
        
        return activations, labels

    def find_selective_neurons(self, layer_idx, threshold=0.8):
        """Find neurons that are selective to specific digits"""
        activations, labels = self.analyze_neuron_activations(layer_idx)
        
        # Get number of channels (neurons) in this layer
        n_neurons = activations.shape[1]
        
        # Calculate mean activation per neuron for each digit
        neuron_digit_activations = defaultdict(dict)
        selective_neurons = {}
        
        for digit in range(10):
            digit_mask = labels == digit
            digit_activations = activations[digit_mask]
            
            # Calculate mean activation per neuron for this digit
            mean_activations = digit_activations.mean(dim=(0, 2, 3))
            
            for neuron_idx in range(n_neurons):
                neuron_digit_activations[neuron_idx][digit] = mean_activations[neuron_idx].item()
        
        # Find selective neurons
        for neuron_idx in range(n_neurons):
            activations_per_digit = np.array([neuron_digit_activations[neuron_idx][d] for d in range(10)])
            max_digit = np.argmax(activations_per_digit)
            selectivity = activations_per_digit[max_digit] / (np.sum(activations_per_digit) + 1e-6)
            
            if selectivity > threshold:
                selective_neurons[neuron_idx] = (max_digit, selectivity)
        
        return selective_neurons, neuron_digit_activations

    def visualize_neuron_preferences(self, layer_idx):
        """Visualize what digits each neuron prefers"""
        _, neuron_activations = self.find_selective_neurons(layer_idx)
        n_neurons = len(neuron_activations)
        
        # Create activation matrix
        activation_matrix = np.zeros((n_neurons, 10))
        for neuron_idx in range(n_neurons):
            for digit in range(10):
                activation_matrix[neuron_idx, digit] = neuron_activations[neuron_idx][digit]
        
        # Normalize activations
        activation_matrix = (activation_matrix - activation_matrix.min()) / (activation_matrix.max() - activation_matrix.min())
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(activation_matrix, xticklabels=range(10), yticklabels=range(n_neurons),
                   cmap='viridis', annot=True, fmt='.2f')
        plt.title(f'Neuron Preferences in Layer {layer_idx+1}')
        plt.xlabel('Digit')
        plt.ylabel('Neuron Index')
        plt.show()

    def visualize_top_activations(self, layer_idx, neuron_idx, top_k=9):
        """Visualize inputs that most activate a specific neuron"""
        activations, labels = self.analyze_neuron_activations(layer_idx)
        
        # Get activation values for specific neuron
        neuron_activations = activations[:, neuron_idx].mean(dim=(1, 2)).numpy()
        
        # Get top-k activating images
        top_indices = np.argsort(neuron_activations)[-top_k:]
        
        # Load original images
        dataset = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())
        
        # Plot top activating images
        plt.figure(figsize=(12, 12))
        for i, idx in enumerate(top_indices):
            plt.subplot(3, 3, i+1)
            plt.imshow(dataset[idx][0].squeeze(), cmap='gray')
            plt.title(f'Digit: {labels[idx]}\nAct: {neuron_activations[idx]:.2f}')
            plt.axis('off')
        plt.suptitle(f'Top {top_k} activations for Neuron {neuron_idx} in Layer {layer_idx+1}')
        plt.show()

    def analyze_neuron_importance(self, layer_idx):
        """Analyze overall importance of neurons in a layer"""
        activations, labels = self.analyze_neuron_activations(layer_idx)
        n_neurons = activations.shape[1]
        
        # Calculate variance of activations for each neuron
        neuron_variance = activations.var(dim=(0, 2, 3)).numpy()
        
        # Plot neuron importance
        plt.figure(figsize=(10, 5))
        plt.bar(range(n_neurons), neuron_variance)
        plt.title(f'Neuron Importance in Layer {layer_idx+1}')
        plt.xlabel('Neuron Index')
        plt.ylabel('Activation Variance')
        plt.show()
        
        return neuron_variance

if __name__ == "__main__":
    analyzer = NeuronAnalyzer('mnist_cnn.pth')
    
    # Analyze each convolutional layer
    for layer_idx in range(4):  # 4 conv layers
        print(f"\nAnalyzing Layer {layer_idx+1}")
        
        # Find selective neurons
        selective_neurons, _ = analyzer.find_selective_neurons(layer_idx)
        print(f"\nSelective neurons in Layer {layer_idx+1}:")
        for neuron, (digit, selectivity) in selective_neurons.items():
            print(f"Neuron {neuron}: Selective for digit {digit} (selectivity: {selectivity:.3f})")
        
        # Visualize neuron preferences
        analyzer.visualize_neuron_preferences(layer_idx)
        
        # Analyze neuron importance
        importance = analyzer.analyze_neuron_importance(layer_idx)
        
        # Visualize top activations for most important neurons
        top_neurons = np.argsort(importance)[-3:]  # Show top 3 neurons
        for neuron_idx in top_neurons:
            analyzer.visualize_top_activations(layer_idx, neuron_idx) 