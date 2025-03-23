import numpy as np
import random
from typing import List, Callable, Union


# =====================
# УТИЛИТЫ ДЛЯ ДАТАСЕТОВ
# =====================

# Базовый класс датасета
class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    # перемешивание датасета
    def shuffle(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(len(self.X))
        self.X = self.X[permutation]
        self.y = self.y[permutation]

    # генерация батчей указанного размера
    def batch(self, batch_size: int):
        for i in range(0, len(self.X), batch_size):
            yield self.X[i:i+batch_size], self.y[i:i+batch_size]

    # применение map-функции к каждому из объектов (x, y)
    def map(self, func: Callable):
        new_X, new_y = [], []
        for x, label in zip(self.X, self.y):
            nx, ny = func(x, label)
            new_X.append(nx)
            new_y.append(ny)
        self.X = np.array(new_X)
        self.y = np.array(new_y)

# Загрузка датасета MNIST из torchvision.
def get_mnist_dataset(train: bool = True) -> Dataset:
    
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError("Для использования MNIST установите PyTorch и torchvision: pip install torch torchvision")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST(root='.', train=train, download=True, transform=transform)

    X = []
    y = []
    for img, label in mnist_data:
        X.append(img.numpy().reshape(-1))
        y.append(label)
    X = np.array(X)
    y = np.array(y)
    return Dataset(X, y)


# ===================================
# СТРУКТУРЫ ДЛЯ ПОСТРОЕНИЯ НЕЙРОСЕТЕЙ
# ===================================

# Слой (базовый класс)
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# Полносвязный слой
class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        limit = 1 / np.sqrt(input_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = np.dot(x, self.W) + self.b
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.dW = np.dot(self.input.T, grad_output)
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.W.T)
        return grad_input

# Функции активации 
class Activation(Layer):
    def __init__(self, activation: Callable, activation_prime: Callable):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.activation_prime(self.input)

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

# Потери (Loss)
class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        raise NotImplementedError

class MSE(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return np.mean((y_pred - y_true)**2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        return 2 * (y_pred - y_true) / y_true.shape[0]

class CrossEntropy(Loss):
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)

        batch_size = y_true.shape[0]
        correct_logprobs = -np.log(probs[range(batch_size), y_true])
        loss = np.mean(correct_logprobs)
        return loss

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        logits = y_pred - np.max(y_pred, axis=1, keepdims=True)
        exps = np.exp(logits)
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        
        batch_size = y_true.shape[0]
        grad = probs
        grad[range(batch_size), y_true] -= 1
        grad /= batch_size
        return grad

# Модель (Network)
class Sequential:
    def __init__(self, layers: List[Layer], loss: Loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def predict(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss_and_grad(self, y_pred: np.ndarray, y_true: np.ndarray):
        loss_value = self.loss.forward(y_pred, y_true)
        grad = self.loss.backward(y_pred, y_true)
        return loss_value, grad


# ============
# ОПТИМИЗАТОРЫ
# ============

class Optimizer:
    def step(self, model: Sequential):
        raise NotImplementedError

# Cтохастический градиентный спуск (SGD)
class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, model: Sequential):
        for layer in model.layers:
            if isinstance(layer, Dense):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

# SGD с momentum
class MomentumSGD(Optimizer):
    def __init__(self, lr: float = 0.01, momentum: float = 0.9):
        self.lr = lr
        self.momentum = momentum
        self.vW = {} 
        self.vb = {}

    def step(self, model: Sequential):
        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                if idx not in self.vW:
                    self.vW[idx] = np.zeros_like(layer.W)
                    self.vb[idx] = np.zeros_like(layer.b)
                self.vW[idx] = self.momentum * self.vW[idx] + self.lr * layer.dW
                self.vb[idx] = self.momentum * self.vb[idx] + self.lr * layer.db

                layer.W -= self.vW[idx]
                layer.b -= self.vb[idx]


# =======
# МЕТРИКИ
# =======

def compute_R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    mean_y_true = np.sum(y_true) / len(y_true)
    ss_total = np.sum((y - mean_y_true) ** 2 for y in y_true)
    ss_residual = np.sum((y_t - y_p) ** 2 for y_t, y_p in zip(y_true, y_pred))
    return (1 - (ss_residual / ss_total))[0]

def compute_MSE(y_pred: np.ndarray, y_test: np.ndarray) -> np.float64:
    return np.mean((y_pred - y_test.reshape(-1,1))**2)
    
def compute_RMSE(y_mse: np.ndarray) -> np.float64:
    return np.sqrt(y_mse)