## CNN Forward Method - PyTorch Deep Learning Implementation

The `forward()` method is the actual network transformation. **The forward method is the mapping that maps an input tensor to a prediction output tensor**.



```python
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

The `relu()` and the `max_pool2d()` calls are just pure operations. Neither of these have weights, and this is why we call them directly from `nn.functional` API, instead of network attributes.

However, in our case, **we won't use `softmax()` because the loss function that we'll use,`F.cross_entropy()`, implicitly performs the `softmax()` operation on its input**, so we'll just return the result of the last linear transformation.
