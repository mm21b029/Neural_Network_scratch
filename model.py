import numpy as np
import wandb
import pickle

class SGDOptimizer:
    def __init__(self, lr, weight_decay, epsilon):
        self.lr = lr
        self.wd = weight_decay
        self.eps = epsilon

    def step(self, weights, biases, grad_w, grad_b):
        for i in range(len(weights)):
            grad_w[i] += self.wd * weights[i]
            weights[i] -= self.lr * grad_w[i]
            biases[i] -= self.lr * grad_b[i]

class MomentumOptimizer(SGDOptimizer):
    def __init__(self, lr, momentum, weight_decay, epsilon):
        super().__init__(lr, weight_decay, epsilon)
        self.momentum = momentum
        self.vel_w = None
        self.vel_b = None

    def step(self, weights, biases, grad_w, grad_b):
        if self.vel_w is None:
            self.vel_w = [np.zeros_like(w) for w in weights]
            self.vel_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            grad_w[i] += self.wd * weights[i]
            self.vel_w[i] = self.momentum * self.vel_w[i] + self.lr * grad_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] + self.lr * grad_b[i]
            weights[i] -= self.vel_w[i]
            biases[i] -= self.vel_b[i]

class NAGOptimizer(MomentumOptimizer):
    def __init__(self, lr, momentum, weight_decay, epsilon):
        super().__init__(lr, momentum, weight_decay, epsilon)
        self.prev_vel_w = None
        self.prev_vel_b = None

    def step(self, weights, biases, grad_w, grad_b):
        if self.prev_vel_w is None:
            self.prev_vel_w = [np.zeros_like(v) for v in self.vel_w]
            self.prev_vel_b = [np.zeros_like(v) for v in self.vel_b]
            
        for i in range(len(weights)):
            self.prev_vel_w[i] = self.vel_w[i].copy()
            self.prev_vel_b[i] = self.vel_b[i].copy()
        
        lookahead_weights = [w + self.momentum * self.vel_w[i] for i, w in enumerate(weights)]
        lookahead_biases = [b + self.momentum * self.vel_b[i] for i, b in enumerate(biases)]
        
        for i in range(len(weights)):
            grad_w[i] += self.wd * lookahead_weights[i]
            grad_b[i] += self.wd * lookahead_biases[i]
            
            self.vel_w[i] = self.momentum * self.vel_w[i] + self.lr * grad_w[i]
            self.vel_b[i] = self.momentum * self.vel_b[i] + self.lr * grad_b[i]
        
        for i in range(len(weights)):
            weights[i] = lookahead_weights[i] - self.vel_w[i]
            biases[i] = lookahead_biases[i] - self.vel_b[i]

class RMSpropOptimizer(SGDOptimizer):
    def __init__(self, lr, beta, weight_decay, epsilon):
        super().__init__(lr, weight_decay, epsilon)
        self.beta = beta
        self.cache_w = None
        self.cache_b = None

    def step(self, weights, biases, grad_w, grad_b):
        if self.cache_w is None:
            self.cache_w = [np.zeros_like(w) for w in weights]
            self.cache_b = [np.zeros_like(b) for b in biases]
        
        for i in range(len(weights)):
            grad_w[i] += self.wd * weights[i]
            self.cache_w[i] = self.beta * self.cache_w[i] + (1 - self.beta) * grad_w[i]**2
            self.cache_b[i] = self.beta * self.cache_b[i] + (1 - self.beta) * grad_b[i]**2
            weights[i] -= self.lr * grad_w[i] / (np.sqrt(self.cache_w[i]) + self.eps)
            biases[i] -= self.lr * grad_b[i] / (np.sqrt(self.cache_b[i]) + self.eps)

class AdamOptimizer(SGDOptimizer):
    def __init__(self, lr, beta1, beta2, weight_decay, epsilon):
        super().__init__(lr, weight_decay, epsilon)
        self.beta1 = beta1
        self.beta2 = beta2
        self.m_w = self.m_b = self.v_w = self.v_b = None
        self.t = 0

    def step(self, weights, biases, grad_w, grad_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1
        
        for i in range(len(weights)):
            grad_w[i] += self.wd * weights[i]
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_w[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b[i]**2
            
            mw_hat = self.m_w[i] / (1 - self.beta1**self.t)
            mb_hat = self.m_b[i] / (1 - self.beta1**self.t)
            vw_hat = self.v_w[i] / (1 - self.beta2**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            weights[i] -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

class NadamOptimizer(AdamOptimizer):
    def step(self, weights, biases, grad_w, grad_b):
        if self.m_w is None:
            self.m_w = [np.zeros_like(w) for w in weights]
            self.m_b = [np.zeros_like(b) for b in biases]
            self.v_w = [np.zeros_like(w) for w in weights]
            self.v_b = [np.zeros_like(b) for b in biases]
        self.t += 1
        
        for i in range(len(weights)):
            grad_w[i] += self.wd * weights[i]
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * grad_w[i]**2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * grad_b[i]**2
            
            mw_hat = (self.beta1 * self.m_w[i] + (1 - self.beta1) * grad_w[i]) / (1 - self.beta1**self.t)
            mb_hat = (self.beta1 * self.m_b[i] + (1 - self.beta1) * grad_b[i]) / (1 - self.beta1**self.t)
            vw_hat = self.v_w[i] / (1 - self.beta2**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)
            
            weights[i] -= self.lr * mw_hat / (np.sqrt(vw_hat) + self.eps)
            biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


class NeuralNetwork:
    def __init__(
        self,
        input_size=784,
        hidden_size=4,
        num_layers=1,
        output_size=10,
        activation="relu",
        loss="cross_entropy",
        weight_init="random",
        optimizer="sgd",
        learning_rate=0.1,
        momentum=0.5,
        beta=0.5,
        beta1=0.5,
        beta2=0.5,
        epsilon=1e-6,
        weight_decay=0.0005,
    ):
        self.activation = activation
        self.loss = loss
        self.weight_init = weight_init
        self.layer_sizes = [input_size] + [hidden_size] * num_layers + [output_size]
        self.weights, self.biases = self._initialize_parameters()
        self.optimizer = self._initialize_optimizer(
            optimizer, learning_rate, momentum, beta, beta1, beta2, epsilon, weight_decay
        )

        # Verify output layer
        assert self.layer_sizes[-1] == 10, \
            f"Output layer must have 10 neurons, got {self.layer_sizes[-1]}"
        
        # print("Model layer sizes:", self.layer_sizes)

    def _initialize_parameters(self):
        weights, biases = [], []
        for i in range(len(self.layer_sizes) - 1):
            in_dim, out_dim = self.layer_sizes[i], self.layer_sizes[i+1]
            if self.weight_init == "xavier":
                if self.activation in ["sigmoid", "tanh"]:
                    std = np.sqrt(2.0 / (in_dim + out_dim))
                elif self.activation == "relu":
                    std = np.sqrt(2.0 / in_dim)
                else:
                    std = 0.01
                weights.append(np.random.normal(0, std, (in_dim, out_dim)))
            else:
                weights.append(np.random.randn(in_dim, out_dim) * 0.01)
            biases.append(np.zeros(out_dim))
        return weights, biases

    def _initialize_optimizer(self, opt_name, lr, momentum, beta, beta1, beta2, eps, wd):
        optimizers = {
            "sgd": SGDOptimizer(lr, wd, eps),
            "momentum": MomentumOptimizer(lr, momentum, wd, eps),
            "nag": NAGOptimizer(lr, momentum, wd, eps),
            "rmsprop": RMSpropOptimizer(lr, beta, wd, eps),
            "adam": AdamOptimizer(lr, beta1, beta2, wd, eps),
            "nadam": NadamOptimizer(lr, beta1, beta2, wd, eps),
        }
        return optimizers[opt_name]
    

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        x = np.atleast_2d(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        elif X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        X = X.astype(np.float32) / 255.0


        activations = [X]

        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                if self.activation == "sigmoid":
                    a = self._sigmoid(z)
                elif self.activation == "tanh":
                    a = self._tanh(z)
                elif self.activation == "relu":
                    a = self._relu(z)
                else:
                    a = z

            activations.append(a)

        return activations

    def backward(self, X, y, activations):
        m = X.shape[0]
        y_one_hot = self._one_hot_encode(y, 10)
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]

        if self.loss == "cross_entropy":
            dZ = activations[-1] - y_one_hot
        else:
            dZ = (activations[-1] - y_one_hot) / m

        for i in reversed(range(len(self.weights))):
            grads_w[i] = np.dot(activations[i].T, dZ) / m
            grads_b[i] = np.sum(dZ, axis=0) / m

            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                if self.activation == "sigmoid":
                    dZ = dA_prev * (activations[i] * (1 - activations[i]))
                elif self.activation == "tanh":
                    dZ = dA_prev * (1 - np.square(activations[i]))
                elif self.activation == "relu":
                    dZ = dA_prev * (activations[i] > 0)
                else:
                    dZ = dA_prev

        self.optimizer.step(self.weights, self.biases, grads_w, grads_b)


    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size):
        train_losses, val_accuracies = [], []
        best_val_acc = 0
        
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            epoch_loss = 0
            
            for i in range(0, X_train.shape[0], batch_size):
                end_idx = min(i + batch_size, len(X_train))
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                X_batch = X_batch.reshape(-1, 784)
                
                activations = self.forward(X_batch)
                output = activations[-1].reshape(-1, 10)

                assert output.shape[1] == 10, \
                    f"Output has {output.shape[1]} classes (expected 10)"

                if self.loss == 'cross_entropy':
                    y_batch = y_batch.astype(int)
                    
                    output = np.clip(output, 1e-15, 1-1e-15)

                    batch_loss = -np.mean(np.log(output[np.arange(len(y_batch)), y_batch]))
                else:
                    batch_loss = np.mean((output - self._one_hot_encode(y_batch, 10))**2)
                
                epoch_loss += batch_loss
                
                self.backward(X_batch, y_batch, activations)
            
            val_pred = self.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save('best_model.pkl')

            wandb.log({
                "epoch": epoch,
                "train_loss": epoch_loss/len(X_train),
                "val_accuracy": val_acc
            })
            
        return train_losses, val_accuracies

    def predict(self, X):
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        probabilities = self.forward(X)
        return np.argmax(probabilities[-1].reshape(-1, 10), axis=1)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'weights': self.weights, 'biases': self.biases}, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        model = NeuralNetwork()
        model.weights = params['weights']
        model.biases = params['biases']
        return model

    def _one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y]