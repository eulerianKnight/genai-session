import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i])
            b = np.zeros((layer_sizes[i + 1], 1))
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history = []
        self.accuracy_history = []

    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)

    def forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        z_values = []

        for i in range(len(self.weights) - 1):
            z = self.weights[i] @ activations[-1] + self.biases[i]
            z_values.append(z)
            activations.append(self.relu(z))

        z = self.weights[-1] @ activations[-1] + self.biases[-1]
        z_values.append(z)
        activations.append(self.softmax(z))

        return activations, z_values

    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        batch_size = x.shape[1]
        activations, z_values = self.forward(x)

        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]

        delta = activations[-1] - y

        for l in range(len(self.weights) - 1, -1, -1):
            dw[l] = delta @ activations[l].T / batch_size
            db[l] = np.sum(delta, axis=1, keepdims=True) / batch_size

            if l > 0:
                delta = (self.weights[l].T @ delta) * self.relu_derivative(z_values[l - 1])

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dw[i]
            self.biases[i] -= learning_rate * db[i]

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int = 32,
              learning_rate: float = 0.01):
        n_samples = X.shape[1]

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(n_samples)
            X_shuffled = X[:, shuffle_idx]
            y_shuffled = y[:, shuffle_idx]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[:, i:i + batch_size]
                y_batch = y_shuffled[:, i:i + batch_size]

                self.backward(X_batch, y_batch, learning_rate)

            _, loss = self.calculate_loss(X, y)
            accuracy = self.calculate_accuracy(X, y)
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

    def calculate_loss(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        activations, _ = self.forward(X)
        predictions = activations[-1]
        loss = -np.sum(y * np.log(predictions + 1e-15)) / X.shape[1]
        return predictions, loss

    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        predictions, _ = self.calculate_loss(X, y)
        predicted_classes = np.argmax(predictions, axis=0)
        true_classes = np.argmax(y, axis=0)
        return np.mean(predicted_classes == true_classes)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions for input X"""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=0)

    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.plot(self.loss_history)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        ax2.plot(self.accuracy_history)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000


    # Generate spiral data
    def generate_spiral_data(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.sqrt(np.random.rand(n_samples)) * 2 * np.pi

        r_a = theta + np.pi
        data_a = np.column_stack([np.cos(r_a) * r_a, np.sin(r_a) * r_a])

        r_b = -theta - np.pi
        data_b = np.column_stack([np.cos(r_b) * r_b, np.sin(r_b) * r_b])

        X = np.vstack([data_a, data_b]).T
        y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

        # Normalize data
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

        # Convert to one-hot encoding
        Y = np.zeros((2, 2 * n_samples))
        Y[0, y == 0] = 1
        Y[1, y == 1] = 1

        return X, Y


    # Generate and plot data
    X, Y = generate_spiral_data(n_samples)

    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], c='blue', label='Class 0')
    plt.scatter(X[0, Y[1] == 1], X[1, Y[1] == 1], c='red', label='Class 1')
    plt.title('Spiral Dataset')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Create and train network
    nn = NeuralNetwork([2, 16, 8, 2])
    nn.train(X, Y, epochs=100, batch_size=32, learning_rate=0.01)

    # Plot training history
    nn.plot_training_history()

    # Visualize decision boundary
    x_min, x_max = X[0].min() - 0.5, X[0].max() + 0.5
    y_min, y_max = X[1].min() - 0.5, X[1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Reshape grid points for prediction
    grid = np.vstack([xx.ravel(), yy.ravel()])

    # Get predictions
    Z = nn.predict(grid).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], c='blue', label='Class 0')
    plt.scatter(X[0, Y[1] == 1], X[1, Y[1] == 1], c='red', label='Class 1')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()