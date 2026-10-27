import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("XOR Neural Network Visualizer")
st.write("Understand how a neural network learns the XOR problem.")

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Initialize weights and biases
W1 = np.array([[5.0, -5.0],
               [5.0, -5.0]])
b1 = np.array([[-2.5, 7.5]])

W2 = np.array([[5.0], [5.0]])
b2 = np.array([[-7.5]])

lr = 0.5

# Train the neural network
for epoch in range(10000):
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    error = output - Y.reshape(-1, 1)

    d_output = error * output * (1 - output)
    d_hidden = np.dot(d_output, W2.T) * hidden * (1 - hidden)

    W2 -= lr * np.dot(hidden.T, d_output) / 4
    b2 -= lr * np.mean(d_output, axis=0, keepdims=True)

    W1 -= lr * np.dot(X.T, d_hidden) / 4
    b1 -= lr * np.mean(d_hidden, axis=0, keepdims=True)

# Test predictions
hidden = sigmoid(np.dot(X, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)
predictions = (output >= 0.5).astype(int).flatten()

st.subheader("XOR Predictions")

for i in range(4):
    st.write(
        f"Input: {X[i].tolist()} | "
        f"Expected: {Y[i]} | "
        f"Predicted: {predictions[i]}"
    )

correct = np.sum(predictions == Y)
st.metric("Correct Predictions", f"{correct}/4")

# Plot decision boundary
st.subheader("Decision Boundary")

xx, yy = np.meshgrid(
    np.linspace(-0.2, 1.2, 100),
    np.linspace(-0.2, 1.2, 100)
)

points = np.c_[xx.ravel(), yy.ravel()]
hidden = sigmoid(np.dot(points, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)
Z = (output >= 0.5).astype(int).reshape(xx.shape)

fig, ax = plt.subplots()
ax.contourf(xx, yy, Z, alpha=0.3)

for i in range(4):
    ax.scatter(X[i][0], X[i][1], s=100)

ax.set_xlabel("Input A")
ax.set_ylabel("Input B")
ax.set_title("XOR Decision Boundary")

st.pyplot(fig)
