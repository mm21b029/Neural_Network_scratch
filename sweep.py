import wandb
from sklearn.model_selection import train_test_split
from keras.datasets import fashion_mnist
import numpy as np
from model import *
import yaml

with open("sweep.cfg", "r") as file:
    sweep_config = yaml.safe_load(file)


sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "max_iter": 10,
    },
    "parameters": {
        "num_layers": {"values": [3, 5]},
        "hidden_size": {"values": [64, 128]},
        "weight_decay": {"values": [0.0005, 0.5]},
        "learning_rate": {"values": [1e-3]},
        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [64]},
        "weight_init": {"values": ["xavier"]},
        "activation": {"values": ["relu"]},
        "epochs": {"values": [5, 10]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="fashion-mnist")

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



def train_model():
    run = wandb.init(project="fashion-mnist")
    
    run_name = f"hl_{run.config.num_layers}_hs_{run.config.hidden_size}_bs_{run.config.batch_size}_ac_{run.config.activation}_wd_{run.config.weight_decay}_lr_{run.config.learning_rate}_opt_{run.config.optimizer}_wi_{run.config.weight_init}_ep_{run.config.epochs}"
    run.name = run_name
    
    config = wandb.config
    
    model = NeuralNetwork(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        activation=config.activation,
        weight_init=config.weight_init
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

    model.train(X_train, y_train, X_val, y_val, epochs=config.epochs, batch_size=config.batch_size)

    test_pred = model.predict(X_test)
    test_acc = np.mean(test_pred == y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    wandb.log({"test_accuracy": test_acc})



wandb.agent(sweep_id, train_model)

wandb.finish()