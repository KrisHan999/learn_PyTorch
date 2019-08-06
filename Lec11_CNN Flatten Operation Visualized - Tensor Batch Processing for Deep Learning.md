## CNN Flatten Operation Visualized - Tensor Batch Processing for Deep Learning

Input tp a convolutional neural network typically have 4 axes:

- (Batch Size, Channels, Height, Width)

#### *Building a tensor representation for a batch of images*

```python
t1 = torch.tensor([
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1],
    [1,1,1,1]
])

t2 = torch.tensor([
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2],
    [2,2,2,2]
])

t3 = torch.tensor([
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3],
    [3,3,3,3]
])

> t = torch.stack((t1, t2, t3))
> t.shape

torch.Size([3, 4, 4])

> t = t.reshape(3,1,4,4)
```

We used the `stack()` function to concatenate our sequence of three tensors along a new axis.

#### `torch.stack()`

`torch.stack(*seq*,*dim=0*,*out=None*)→ Tensor`

- Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.

- Parameters

  - **seq**(*sequence of Tensors*) – sequence of tensors to concatenate

  - **dim**([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)")) – dimension to insert. Has to be between 0 and the number of dimensions of concatenated tensors (inclusive)

  - **out**([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")*,**optional*) – the output tensor

#### Flattening specific axes of a tensor

#### `torch.flatten()`

```python
# t -> (3, 1, 4, 4)
> t.flatten(start_dim=1).shape
torch.Size([3, 16])

> t.flatten(start_dim=1)
tensor(
[
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
]
)
```
