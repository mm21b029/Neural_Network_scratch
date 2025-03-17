import wandb
from sklearn.model_selection import train_test_split
import numpy as np
from model import *
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train a neural network with W&B sweeps")
    
    parser.add_argument("-wp", "--wandb_project", type=str, default="fashion-mnist_mm21b029", help="Project name for W&B")
    parser.add_argument("-we", "--wandb_entity", type=str, default="hemachandra0801-iit-madras", help="W&B Entity")
    parser.add_argument("-d", "--dataset", type=str, choices=["mnist", "fashion_mnist"], default="fashion_mnist", help="Dataset")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("-l", "--loss", type=str, choices=["MSE", "cross-entropy"], default="cross-entropy", help="Loss function")
    parser.add_argument("-o", "--optimizer", type=str, choices=["SGD", "momentum", "nesterov", "RMSprop", "Adam"], default="RMSprop", help="Optimizer")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("-m", "--momentum", type=float, default=0, help="Momentum for optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.9, help="Beta for rmsprop")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9, help="Beta1 for adam/nadam")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999, help="Beta2 for adam/nadam")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon for optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("-w_i", "--weight_init", type=str, choices=["random", "xavier"], default="xavier", help="Weight initialization")
    parser.add_argument("-nhl", "--num_layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Size of hidden layers")
    parser.add_argument("-a", "--activation", type=str, choices=["sigmoid", "tanh", "relu"], default="tanh", help="Activation function")
    
    return parser.parse_args()

def load_data(dataset):
    if dataset == "fashion_mnist":
        from keras.datasets import fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    elif dataset == "mnist":
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def train():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()
    
    wandb.init(project="fashion-mnist")
    model = NeuralNetwork(**wandb.config)
    
    model.train(X_train, y_train, X_val, y_val, epochs=wandb.config.epochs, batch_size=wandb.config.batch_size)
    
    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})


def visualize_dataset():
    import matplotlib.pyplot as plt
    from keras.datasets import fashion_mnist

    (x_train, y_train), (_, _), (_, _) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    indices = [np.where(y_train == i)[0][0] for i in range(10)]

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_train[indices[i]], cmap='gray_r')
        plt.title(class_names[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    args = get_args()

    # visualize_dataset()

    wandb.init(
        project=args.wandb_project, 
        name="Run", 
        entity=args.wandb_entity,
        config={"epochs": args.epochs, "batch_size": args.batch_size}
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(args.dataset)

    model = NeuralNetwork(
        num_layers=args.num_layers,
        hidden_size=[args.hidden_size]*args.num_layers,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        activation=args.activation,
        weight_init=args.weight_init,
        loss=args.loss,
        beta=args.beta,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
    )

    model.train(X_train, y_train, X_val, y_val, epochs=args.epochs, batch_size=args.batch_size)

    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    wandb.log({"accuracy": test_acc})

    wandb.finish()