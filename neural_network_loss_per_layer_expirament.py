import datetime
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_random_seeds(seed_value=42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_data_loader(batch_size=64, dataset="MNIST"):
    if dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_set = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='../data', train=False, download=True, transform=transform)
        input_channels, input_size = 1, 28 * 28
    elif dataset == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        input_channels, input_size = 3, 32 * 32 * 3
    else:
        raise ValueError("Dataset not supported yet")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True, pin_memory_device="cuda")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, persistent_workers=True, pin_memory=True, pin_memory_device="cuda")
    return train_loader, test_loader, input_channels, input_size


class LayerWithLoss(nn.Module):
    def __init__(self, layer, layer_index):
        super().__init__()
        self.layer = layer
        self.layer_index = layer_index
        self.learning_curve = []

    def forward(self, x):
        output = self.layer(x)
        self.learning_curve.append(output.detach().mean().item())
        return output


def create_fc_network(input_size, num_layers, hidden_size, output_size):
    layers = []
    sizes = [input_size] + [hidden_size] * (num_layers - 1) + [output_size]
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
    return DynamicNetwork(layers, is_cnn=False)


class DynamicNetwork(nn.Module):
    def __init__(self, layers, is_cnn=False):
        super().__init__()
        self.layers = nn.ModuleList([LayerWithLoss(layer, i) for i, layer in enumerate(layers)])
        self.is_cnn = is_cnn
        self.gradient_norms = []
        self.layer_outputs = []
        self.isolated_grads = {}

    def forward(self, x, target, experiment_type, current_epoch, switch_epoch):
        self.layer_outputs = []

        if not self.is_cnn:
            x = x.view(x.size(0), -1)  # Flatten the input for FC networks

        for i, layer in enumerate(self.layers):
            x = layer(x)
            self.layer_outputs.append(x)

            if experiment_type == "isolated_after_switch" and current_epoch >= switch_epoch:
                x = self._isolate_layer(x, i)
            elif experiment_type == "always_isolated":
                x = self._isolate_layer(x, i)

        return x, self.layer_outputs

    def _isolate_layer(self, x, layer_index):
        x_detached = x.detach()
        x_isolated = x_detached.requires_grad_()

        def hook(grad):
            self.isolated_grads[layer_index] = grad.clone()
            return grad

        x_isolated.register_hook(hook)
        return x_isolated

    def apply_isolated_gradients(self):
        for i, layer in enumerate(self.layers):
            if i in self.isolated_grads:
                if hasattr(layer.layer, 'weight') and layer.layer.weight.grad is not None:
                    layer.layer.weight.grad.data.copy_(self.isolated_grads[i])
                if hasattr(layer.layer, 'bias') and layer.layer.bias.grad is not None:
                    layer.layer.bias.grad.data.copy_(self.isolated_grads[i])
        self.isolated_grads.clear()

    def compute_gradient_norms(self):
        total_norm = 0
        for i, p in enumerate(self.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.gradient_norms.append(total_norm)

    def verify_gradient_flow(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer.layer, 'weight') and layer.layer.weight.grad is not None:
                assert torch.isfinite(layer.layer.weight.grad).all(), f"Layer {i} has non-finite gradients"
            if hasattr(layer.layer, 'bias') and layer.layer.bias.grad is not None:
                assert torch.isfinite(layer.layer.bias.grad).all(), f"Layer {i} bias has non-finite gradients"


def create_cnn(input_channels, num_layers, num_classes):
    layers = []
    channels = [input_channels] + [32 * (2 ** i) for i in range(num_layers - 1)]

    for i in range(num_layers - 1):
        layers.extend([
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ])

    layers.append(nn.Flatten())

    with torch.no_grad():
        x = torch.randn(1, input_channels, 28, 28)
        for layer in layers[:-1]:  # Exclude the Flatten layer
            x = layer(x)
        flatten_size = x.view(1, -1).size(1)

    layers.extend([
        nn.Linear(flatten_size, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    ])

    return DynamicNetwork(layers, is_cnn=True)


def get_layer_type(layer):
    if isinstance(layer.layer, nn.Linear):
        return "FC"
    elif isinstance(layer.layer, nn.Conv2d):
        return "Conv2D"
    elif isinstance(layer.layer, nn.ReLU):
        return "ReLU"
    elif isinstance(layer.layer, nn.MaxPool2d):
        return "MaxPool2D"
    elif isinstance(layer.layer, nn.Dropout):
        return "Dropout"
    else:
        return type(layer.layer).__name__


def calculate_loss(outputs, target, criterion, is_intermediate=False, layer_index=None, num_layers=None):
    if is_intermediate:
        constant_target = torch.full_like(outputs, 0.5)
        layer_weight = np.exp(-layer_index / num_layers)
        layer_weight /= num_layers  # Normalize
        return layer_weight * nn.MSELoss()(outputs, constant_target)
    else:
        return criterion(outputs, target)


def moving_average(data, window_size):
    """Computes a moving average with the specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def train_model(model, train_loader, test_loader, optimizer, scheduler, num_epochs, experiment_type, switch_epoch):
    criterion = nn.CrossEntropyLoss()
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in tqdm(range(num_epochs), desc="Epochs Progress", ncols=100):  # Using tqdm for progress monitoring
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False, ncols=100):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs, layer_outputs = model(images, labels, experiment_type, epoch, switch_epoch)

            loss = calculate_loss(outputs, labels, criterion)
            if experiment_type == "per_layer_loss":
                num_layers = len(layer_outputs)
                for i, layer_output in enumerate(layer_outputs[:-1]):
                    loss += calculate_loss(layer_output, labels, criterion, is_intermediate=True, layer_index=i, num_layers=num_layers)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if experiment_type in ["isolated_after_switch", "always_isolated"]:
                model.apply_isolated_gradients()

            model.verify_gradient_flow()
            model.compute_gradient_norms()
            optimizer.step()
            train_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_accuracy = 100 * correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images, labels, "standard", epoch, switch_epoch)
                loss = calculate_loss(outputs, labels, criterion)
                test_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracy = 100 * correct_test / total_test
        test_accuracies.append(test_accuracy)
        scheduler.step()
    return train_losses, test_losses, train_accuracies, test_accuracies


def run_experiment(model_creator, train_loader, test_loader, experiment_type, num_epochs, switch_epoch, num_runs=5):
    all_train_losses = []
    all_test_losses = []
    all_train_accuracies = []
    all_test_accuracies = []
    all_gradient_norms = []

    for run in range(num_runs):
        set_random_seeds(seed_value=42 + run)
        model = model_creator().to(device)
        print(f"Number of trainable parameters: {count_parameters(model)}")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        train_loss, test_loss, train_acc, test_acc = train_model(
            model, train_loader, test_loader, optimizer, scheduler,
            num_epochs, experiment_type, switch_epoch
        )
        all_train_losses.append(train_loss)
        all_test_losses.append(test_loss)
        all_train_accuracies.append(train_acc)
        all_test_accuracies.append(test_acc)
        all_gradient_norms.append(model.gradient_norms)

    # Compute mean and std
    mean_train_loss = np.mean(all_train_losses, axis=0)
    std_train_loss = np.std(all_train_losses, axis=0)
    mean_test_loss = np.mean(all_test_losses, axis=0)
    std_test_loss = np.std(all_test_losses, axis=0)
    mean_train_acc = np.mean(all_train_accuracies, axis=0)
    std_train_acc = np.std(all_train_accuracies, axis=0)
    mean_test_acc = np.mean(all_test_accuracies, axis=0)
    std_test_acc = np.std(all_test_accuracies, axis=0)
    mean_gradient_norms = np.mean(all_gradient_norms, axis=0)
    std_gradient_norms = np.std(all_gradient_norms, axis=0)

    return model, (mean_train_loss, std_train_loss), (mean_test_loss, std_test_loss), (mean_train_acc, std_train_acc), (mean_test_acc, std_test_acc), (mean_gradient_norms, std_gradient_norms)


def create_visualization_folders():
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join('visualizations', timestamp)
    os.makedirs(run_folder)
    return run_folder


def save_plot(plt, filename, run_folder):
    filepath = os.path.join(run_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Plot saved: {filepath}")


def remove_outliers(data, threshold=3):
    """Remove outliers that are more than 'threshold' standard deviations away."""
    mean = np.mean(data)
    std = np.std(data)
    return [x if abs(x - mean) <= threshold * std else mean for x in data]


def plot_combined_gradient_norms(gradient_norms_dict, title="Combined Gradient Norms", model_type="", run_folder="", window_size=10):
    plt.figure(figsize=(12, 6))

    for exp_name, grad_norms in gradient_norms_dict.items():
        mean_grad_norms = grad_norms['mean']
        std_grad_norms = grad_norms['std']

        # Applying moving average smoothing to gradient norms
        if len(mean_grad_norms) >= window_size:
            smoothed_mean = moving_average(mean_grad_norms, window_size)
            smoothed_std = moving_average(std_grad_norms, window_size)
        else:
            smoothed_mean = mean_grad_norms
            smoothed_std = std_grad_norms

        iterations = range(len(smoothed_mean))
        plt.fill_between(iterations, smoothed_mean - smoothed_std, smoothed_mean + smoothed_std, alpha=0.1)
        plt.plot(iterations, smoothed_mean, label=exp_name)

    plt.title(f"{title} ({model_type})")
    plt.xlabel("Iteration")
    plt.ylabel("Gradient Norm")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)

    filename = f"{model_type.replace(' ', '_')}_gradient_norms.png"
    save_plot(plt, filename, run_folder)


def plot_combined_loss(loss_dict, title="Combined Loss Curves", model_type="", run_folder=""):
    plt.figure(figsize=(12, 6))

    all_losses = []
    for exp_name, losses in loss_dict.items():
        all_losses += list(remove_outliers(losses['mean_train'])) + list(remove_outliers(losses['mean_test']))


    for exp_name, losses in loss_dict.items():
        mean_train = remove_outliers(losses['mean_train'])
        std_train = losses['std_train']
        mean_test = remove_outliers(losses['mean_test'])
        std_test = losses['std_test']
        epochs = range(1, len(mean_train) + 1)

        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.1)
        plt.plot(epochs, mean_train, label=f'{exp_name} Train')

        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.1)
        plt.plot(epochs, mean_test, linestyle='--', label=f'{exp_name} Test')

    plt.title(f"{title} ({model_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale('log')  # Set to log scale
    plt.legend()
    plt.grid(True)
    save_plot(plt, f"{model_type.replace(' ', '_')}_loss_curves.png", run_folder)

def plot_combined_accuracy(acc_dict, title="Combined Accuracy Curves", model_type="", run_folder=""):
    plt.figure(figsize=(12, 6))

    all_accuracies = []
    for exp_name, accs in acc_dict.items():
        all_accuracies += list(remove_outliers(accs['mean_train'])) + list(remove_outliers(accs['mean_test']))

    # Create plots with log scale
    for exp_name, accs in acc_dict.items():
        mean_train = remove_outliers(accs['mean_train'])
        std_train = accs['std_train']
        mean_test = remove_outliers(accs['mean_test'])
        std_test = accs['std_test']
        epochs = range(1, len(mean_train) + 1)

        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.1)
        plt.plot(epochs, mean_train, label=f'{exp_name} Train')

        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.1)
        plt.plot(epochs, mean_test, linestyle='--', label=f'{exp_name} Test')

    plt.title(f"{title} ({model_type})")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.yscale('log')  # Set to log scale
    plt.legend()
    plt.grid(True)
    save_plot(plt, f"{model_type.replace(' ', '_')}_accuracy_curves.png", run_folder)


def plot_layer_learning_curves(model, title="Layer Learning Curves", model_type="", run_folder="", window_size=10):
    plt.figure(figsize=(12, 6))
    legend_labels = []

    for i, layer in enumerate(model.layers[:-1]):  # Exclude the last layer
        if hasattr(layer, 'learning_curve'):
            learning_curve = remove_outliers(layer.learning_curve)  # Remove outliers
            if len(learning_curve) >= window_size:
                smoothed_curve = moving_average(learning_curve, window_size)
            else:
                smoothed_curve = learning_curve

            layer_type = get_layer_type(layer)
            if layer_type not in ["ReLU", "Dropout", "MaxPool2D"]:
                plt.plot(smoothed_curve)
                legend_labels.append(f'Layer {i + 1} ({layer_type})')

    plt.title(f"{title} ({model_type})")
    plt.xlabel("Iteration")
    plt.ylabel("Average Layer Output")
    plt.legend(legend_labels)
    plt.grid(True)

    filename = f"{model_type.replace(' ', '_')}_layer_learning_curves.png"
    save_plot(plt, filename, run_folder)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    set_random_seeds()

    run_folder = create_visualization_folders()

    # Configuration
    dataset = "MNIST"
    epochs = 5
    switch_epoch = 2
    batch_size = 512
    layers = 3
    hidden_size = 256
    window_size = max(1, int(epochs * .1))
    num_runs = 1

    # Load dataset
    train_loader, test_loader, input_channels, input_size = get_data_loader(batch_size=batch_size, dataset=dataset)
    num_classes = 10

    experiments = [
        ("standard", "Standard NN"),
        ("isolated_after_switch", "Self-Isolated NN"),
        #("always_isolated", "Always Isolated NN"),
        ("per_layer_loss", "Per-Layer Loss NN")
    ]

    # Define model creators
    fc_creator = lambda: create_fc_network(input_size, layers, hidden_size, num_classes)
    cnn_creator = lambda: create_cnn(input_channels, layers, num_classes)

    fc_grad_norms_dict = {}
    fc_loss_dict = {}
    fc_accuracy_dict = {}
    cnn_grad_norms_dict = {}
    cnn_loss_dict = {}
    cnn_accuracy_dict = {}

    for exp_type, exp_name in experiments:
        print(f"\nRunning {exp_name} experiment...")

        # Fully Connected Network
        print("Fully Connected Network:")
        fc_model, fc_train_loss, fc_test_loss, fc_train_acc, fc_test_acc, fc_grad_norms = run_experiment(
            fc_creator, train_loader, test_loader, exp_type, epochs, switch_epoch, num_runs
        )

        fc_grad_norms_dict[exp_name] = {'mean': fc_grad_norms[0], 'std': fc_grad_norms[1]}
        fc_loss_dict[exp_name] = {'mean_train': fc_train_loss[0], 'std_train': fc_train_loss[1],
                                  'mean_test': fc_test_loss[0], 'std_test': fc_test_loss[1]}
        fc_accuracy_dict[exp_name] = {'mean_train': fc_train_acc[0], 'std_train': fc_train_acc[1],
                                      'mean_test': fc_test_acc[0], 'std_test': fc_test_acc[1]}
        plot_layer_learning_curves(fc_model, f"{exp_name} Layer Learning Curves", "NN", run_folder)

        # CNN
        print("Convolutional Neural Network:")
        cnn_model, cnn_train_loss, cnn_test_loss, cnn_train_acc, cnn_test_acc, cnn_grad_norms = run_experiment(
            cnn_creator, train_loader, test_loader, exp_type, epochs, switch_epoch, num_runs
        )

        cnn_grad_norms_dict[exp_name] = {'mean': cnn_grad_norms[0], 'std': cnn_grad_norms[1]}
        cnn_loss_dict[exp_name] = {'mean_train': cnn_train_loss[0], 'std_train': cnn_train_loss[1],
                                   'mean_test': cnn_test_loss[0], 'std_test': cnn_test_loss[1]}
        cnn_accuracy_dict[exp_name] = {'mean_train': cnn_train_acc[0], 'std_train': cnn_train_acc[1],
                                       'mean_test': cnn_test_acc[0], 'std_test': cnn_test_acc[1]}
        plot_layer_learning_curves(cnn_model, f"{exp_name} Layer Learning Curves", "CNN", run_folder)

    # Plot results for Fully Connected Network
    plot_combined_gradient_norms(fc_grad_norms_dict, "Gradient Norms", "NN", run_folder)
    plot_combined_loss(fc_loss_dict, "Loss Curves", "NN", run_folder)
    plot_combined_accuracy(fc_accuracy_dict, "Accuracy Curves", "NN", run_folder)

    # Plot results for CNN
    plot_combined_gradient_norms(cnn_grad_norms_dict, "Gradient Norms", "CNN", run_folder)
    plot_combined_loss(cnn_loss_dict, "Loss Curves", "CNN", run_folder)
    plot_combined_accuracy(cnn_accuracy_dict, "Accuracy Curves", "CNN", run_folder)

    print(f"All plots have been saved in the folder: {run_folder}")
