## CNN Weights - Learnable Parameters in PyTorch Neural Networks

```python
network = Network()
```

### Learnable Parameters

*Learnable parameters* are parameters whose values are learned during the training process.

The **learnable parameters** are the **weights** inside our network and they live inside each layer.

PyTorch has a special class called `Parameter`. The `Parameter` class extends the tensor class, and so the **weight tensor inside every layer is an instance of this `Parameter` class**.

**PyTorch's `nn.Module` class is basically looking for any attributes whose values are instances of the Paramter class, and when it finds an instance of the parameter class, it keeps track of it.**

### Accessing the Networks Parameters

```python
# get all the learnable parameters inside the network:
network.parameters()
```



```python
for param in network.parameters():
    print(param.shape)

torch.Size([6, 1, 5, 5])
torch.Size([6])
torch.Size([12, 6, 5, 5])
torch.Size([12])
torch.Size([120, 192])
torch.Size([120])
torch.Size([60, 120])
torch.Size([60])
torch.Size([10, 60])
torch.Size([10])
```

```python
for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)

conv1.weight          torch.Size([6, 1, 5, 5])
conv1.bias          torch.Size([6])
conv2.weight          torch.Size([12, 6, 5, 5])
conv2.bias          torch.Size([12])
fc1.weight          torch.Size([120, 192])
fc1.bias          torch.Size([120])
fc2.weight          torch.Size([60, 120])
fc2.bias          torch.Size([60])
out.weight          torch.Size([10, 60])
out.bias          torch.Size([10])
```


