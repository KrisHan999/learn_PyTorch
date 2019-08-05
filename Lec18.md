## CNN Layers - PyTorch Deep Neural Network Architecture

### Our CNN Layers

```python
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        return t
```

Each of layers extends PyTorch's neural network `Module` class. For each layer, there are **two primary items** encapsulated inside, **a forward function definition and a weight tensor**.

As we specify our **layers as attributes** inside our `Network` class, PyTorch's neural network `Module` class keeps track of the **weight tensors** inside each layer. 

All we have to do is **assign our layers as attributes** inside our network module, and **the `Module` base class will see this and register the weights as learnable paramters of our networks.**

### CNN Layer Parameters

#### *Parameter vs Argument*

We'll **parameters** are used in function definitions as **place-holders** while **arguments** are the **actual values** that are passed to the function.

#### *Two types of parameters*

1. Hyperparameters

2. Data dependent hyperparameters


