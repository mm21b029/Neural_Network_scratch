# Neural Network Implementation from scratch

WandB report link: https://wandb.ai/hemachandra0801-iit-madras/fashion-mnist/reports/DA6401-Assignment-1--VmlldzoxMTg0MTM0Ng

## Installation

Before running the project, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

#### Example (Using Fashion MNIST):

```python
from keras.datasets import fashion_mnist
import numpy as np

def load_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
```

---

### 2. Initialize, Train and Validate the Model

```python
import numpy as np
from model import Model  # Ensure model.py is in the same directory

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
```

---

## Hyperparameter Tuning with WandB

### A. Running a Sweep using `sweep.py`

#### 1. Configure the Sweep
Modify `sweep.cfg` to specify the hyperparameters for tuning.

#### 2. Run the Sweep
```bash
python sweep.py
```
This will initialize and run the hyperparameter tuning process using WandB.

---

### B. Running `train.py` from Command Line

To train the model manually, use `train.py` with the following command-line arguments.

#### Accepted Arguments
The table below describes the available arguments, their accepted values, and default settings:

| Tag | Argument | Accepted Values | Default |
|------|----------|----------------|---------|
| `-wp` | `--wandb_project` | Any string | `DL` |
| `-we` | `--wandb_entity` | Any string | `mm21b030-indian-institute-of-technology-madras` |
| `-d` | `--dataset` | `mnist`, `fashion_mnist` | `fashion_mnist` |
| `-e` | `--epochs` | Any integer | `10` |
| `-b` | `--batch_size` | Any integer | `128` |
| `-l` | `--loss` | `MSE`, `cross-entropy` | `cross-entropy` |
| `-o` | `--optimizer` | `SGD`, `momentum`, `nesterov`, `RMSprop`, `Adam` | `RMSprop` |
| `-lr` | `--learning_rate` | Any float | `0.001` |
| `-m` | `--momentum` | Any float | `0` |
| `-beta` | `--beta` | Any float | `0.9` |
| `-beta1` | `--beta1` | Any float | `0.9` |
| `-beta2` | `--beta2` | Any float | `0.999` |
| `-eps` | `--epsilon` | Any float | `0.000001` |
| `-w_d` | `--weight_decay` | Any float | `0.0005` |
| `-w_i` | `--weight_init` | `random`, `xavier` | `xavier` |
| `-nhl` | `--num_layers` | Any integer | `3` |
| `-sz` | `--hidden_size` | Any integer | `128` |
| `-a` | `--activation` | `sigmoid`, `tanh`, `relu` | `tanh` |

#### Example Usage
Run training with custom parameters:
```bash
python train.py -e 20 -b 64 -lr 0.0005 -o Adam -a relu
```

For more details, use:
```bash
python train.py --help
```