import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neural_network import MLPClassifier

# Create folder for generated plots

os.makedirs("plots", exist_ok=True)

# A1: Basic Functions

def summation(x, w, b):
    return np.dot(x, w) + b

def step(x):
    return 1 if x >= 0 else 0

def bipolar(x):
    return 1 if x >= 0 else -1

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return max(0, x)

def leaky_relu(x):
    return x if x > 0 else 0.01 * x

# Perceptron

def train(X, y, w, b, lr, activation, epochs=1000):
    errors = []

    for epoch in range(epochs):
        total_error = 0

        for x, target in zip(X, y):
            net = summation(x, w, b)
            out = activation(net)
            error = target - out

            w += lr * error * x
            b += lr * error
            total_error += error ** 2

        errors.append(total_error)

        if total_error <= 0.002:
            break

    return w, b, epoch + 1, errors


def predict(X, w, b, activation):
    return np.array([
        activation(summation(x, w, b))
        for x in X
    ])

# AND and XOR Data

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

AND = np.array([0, 0, 0, 1])
XOR = np.array([0, 1, 1, 0])

# A2: AND Gate

w = np.array([0.2, -0.75], dtype=float)
b = 10
lr = 0.05

w_and, b_and, ep_and, err_and = train(
    X, AND, w, b, lr, step
)

print("AND weights:", w_and)
print("AND bias:", b_and)
print("AND epochs:", ep_and)
print("AND output:", predict(X, w_and, b_and, step))

plt.figure()
plt.plot(err_and, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("AND Gate Error")
plt.grid(True)
plt.savefig("plots/and_gate_error.png")
plt.show()

# A3: Activation Comparison

activations = {
    "Step": step,
    "Bipolar": bipolar,
    "Sigmoid": sigmoid,
    "ReLU": relu
}

activation_epochs = {}

for name, func in activations.items():

    target = np.array([-1, -1, -1, 1]) if name == "Bipolar" else AND

    w, b, epochs, errors = train(
        X,
        target,
        np.array([0.2, -0.75], dtype=float),
        10,
        0.05,
        func
    )

    activation_epochs[name] = epochs
    print(name, ":", epochs, "epochs")

plt.figure()
plt.bar(
    activation_epochs.keys(),
    activation_epochs.values()
)
plt.xlabel("Activation Function")
plt.ylabel("Iterations")
plt.title("Activation Function vs Iterations")
plt.grid(axis='y')
plt.savefig("plots/activation_comparison.png")
plt.show()

# A4: Learning Rate Comparison

rates = np.arange(0.1, 1.1, 0.1)
iterations = []

for rate in rates:

    w, b, epochs, errors = train(
        X,
        AND,
        np.array([0.2, -0.75], dtype=float),
        10,
        rate,
        step
    )

    iterations.append(epochs)

plt.figure()
plt.plot(rates, iterations, marker='o')
plt.xlabel("Learning Rate")
plt.ylabel("Iterations")
plt.title("Learning Rate vs Iterations")
plt.grid(True)
plt.savefig("plots/learning_rate_vs_iterations.png")
plt.show()

# A5: XOR

w_xor, b_xor, ep_xor, err_xor = train(
    X,
    XOR,
    np.array([0.2, -0.75], dtype=float),
    10,
    0.05,
    step
)

print("XOR epochs:", ep_xor)
print("XOR output:", predict(X, w_xor, b_xor, step))

plt.figure()
plt.plot(err_xor, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("XOR Gate Error")
plt.grid(True)
plt.savefig("plots/xor_gate_error.png")
plt.show()

# A6: Customer Data

customers = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198]
], dtype=float)

y = np.array([
    1, 1, 1, 0, 1,
    0, 1, 1, 0, 0
])

customers = (
    customers - customers.min(axis=0)
) / (
    customers.max(axis=0) - customers.min(axis=0)
)

w, b, epochs, errors = train(
    customers,
    y,
    np.ones(4) * 0.1,
    0.1,
    0.05,
    sigmoid
)

prob = predict(customers, w, b, sigmoid)
pred = (prob >= 0.5).astype(int)

print("Customer predictions:", pred)
print("Actual:", y)
print("Customer epochs:", epochs)

# A7: Pseudo-Inverse

Xb = np.c_[
    np.ones(len(customers)),
    customers
]

pinv_w = np.linalg.pinv(Xb) @ y
pinv_pred = (Xb @ pinv_w >= 0.5).astype(int)

print("Pseudo-inverse predictions:", pinv_pred)

# A8: MLP for AND

mlp_and = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    learning_rate_init=0.05,
    max_iter=1000,
    random_state=1
)

mlp_and.fit(X, AND)

print("MLP AND:", mlp_and.predict(X))
print("MLP AND iterations:", mlp_and.n_iter_)

plt.figure()
plt.plot(mlp_and.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP AND Gate Training")
plt.grid(True)
plt.savefig("plots/mlp_and_training.png")
plt.show()

# A9: MLP for XOR

mlp_xor = MLPClassifier(
    hidden_layer_sizes=(2,),
    activation='logistic',
    learning_rate_init=0.05,
    max_iter=5000,
    random_state=1
)

mlp_xor.fit(X, XOR)

print("MLP XOR:", mlp_xor.predict(X))
print("MLP XOR iterations:", mlp_xor.n_iter_)

plt.figure()
plt.plot(mlp_xor.loss_curve_)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("MLP XOR Training")
plt.grid(True)
plt.savefig("plots/mlp_xor_training.png")
plt.show()

# A10: Two Output Nodes

two_output = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1]
])

print("Two-output targets:")
print(two_output)


# A11: MLPClassifier

print("AND MLP iterations:", mlp_and.n_iter_)
print("XOR MLP iterations:", mlp_xor.n_iter_)