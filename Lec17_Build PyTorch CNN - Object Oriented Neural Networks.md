## Build PyTorch CNN - Object Oriented Neural Networks

#### *Prerequisites*

To build neural networks in PyTorch, we extend the `torch.nn.Module` PyTorch class. It is the **base class** for all of neural network modules which includes layers.

#### *PyTorch's `nn.Module` class*

**All of the layers** in PyTorch **extend** the `nn.Module` class and **inherit** all of PyTorch's built-in functionality within the `nn.Module` class.

Even **neural networks extend** the `nn.Module` class. This makes sense because neural networks can be thought of as one big layer.

This means we must extend `nn.Module` class when building a new layer or neural network.

#### *PyTorch `nn.Module` have a `forward()` method*

> The tensor input is passed forward through the network

Every PyTorch `nn.Module` has a `forward` method, and so when we are building layers and networks, we must **provide an implementation** of the `forward()` method. The `forward` method is the actual transformation.

#### *PyTorch's `nn.functional` package*

This package provides us many neural network operations that we can use for building layers.

### Building a neural network in PyTorch

1. Create a neural network class that extend network's layers as class base class.

2. In the class constructor, define the network's layers as class attributes using pre-built layers from `torch.nn`

3. Use the network's layer attributes as well as operations from the `nn.functional` API to define the network's forward pass.

```python
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        # size of in_features of the first fc is determined by the previous layers and operations.
        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # (1) input layer
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12 * 4 * 4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)

        return t
```
