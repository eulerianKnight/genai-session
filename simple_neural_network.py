import numpy as np
import matplotlib.pyplot as plt


# Activation Functions and Derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Generate Dummy Data
np.random.seed(42)
x = np.linspace(-1, 1, 100).reshape(-1, 1)
y = np.sin(3 * x) + np.random.normal(0, 0.1, x.shape)  # Noisy sine wave

# Initialize Weights
input_size, hidden_size, output_size = 1, 10, 1
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training Parameters
epochs = 5000
learning_rate = 0.01
losses = []

# Training Loop
for epoch in range(epochs):
    # Forward Pass
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = z2  # Linear output

    # Compute Loss (Mean Squared Error)
    loss = np.mean((y - y_pred) ** 2)
    losses.append(loss)

    # Backpropagation
    error = y_pred - y
    dW2 = np.dot(a1.T, error) / len(x)
    db2 = np.mean(error, axis=0)

    hidden_error = np.dot(error, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(x.T, hidden_error) / len(x)
    db1 = np.mean(hidden_error, axis=0)

    # Update Weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # Print Progress
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.5f}')

# Plot Loss Curve
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plot Predictions
plt.scatter(x, y, label='True Data')
plt.plot(x, y_pred, color='red', label='Predicted')
plt.legend()
plt.title('Neural Network Regression')
plt.show()