## CNN Image Prediction with PyTorch - Forward Propagation Explained

### Predicting with the network: Forward pass

Turn off PyTorch’s gradient calculation feature since we don't need to update the network's weights. 

```python
> torch.set_grad_enabled(False) 
<torch.autograd.grad_mode.set_grad_enabled at 0x17c4867dcc0>
```

The **computation graph** keeps track of the network's mapping by tracking each computation that happens. The graph is **used** during the training process to **calculate the derivative** (gradient) of the loss function with respect to the network’s weights.

**Turning it off** isn’t strictly necessary but having the feature turned off does **reduce memory consumption since the graph isn't stored in memory**.

### Predict with the Network object instead of the forward() function

```python
> network = Network()

> sample = next(iter(train_set)) 
> image, label = sample 
> image.shape 
torch.Size([1, 28, 28])

> pred = network(image.unsqueeze(0)) # image shape needs to be (batch_size × in_channels × H × W)

> pred
tensor([[0.0991, 0.0916, 0.0907, 0.0949, 0.1013, 0.0922, 0.0990, 0.1130, 0.1107, 0.1074]])

> pred.shape
torch.Size([1, 10])

> label
9

> pred.argmax(dim=1)
tensor([7])
```

### Input data should map the format of the network's input

For 2d convolutional network, the input data format should be like: (batch, channel, height, width)
