## Flatten, Reshape, and Squeeze Explained - Tensor for Deep Learning with PyTroch

### Tensor operation types

High-level categories of operations:

- Reshsaping operations

- Element-wise operations

- Reduction operations

- Access operations

### Reshaping operations for tensors

In PyTorch, we have two ways to get the shape:

```python
> t.size()
torch.Size([3, 4])

> t.shape
torch.Size([3, 4])
```

We can get the number of elements in the tensor by:

```python
> t.numel()
12
```

**Reshaping changes the tensor's shape but not the underlying data.**

#### *Reshaping a tensor in PyTorch*

- `reshape()`

- `view()`

```python
t = torch.tensor([

    [1,1,1,1],

    [2,2,2,2],

    [3,3,3,3]

], dtype=torch.float32)

t.reshape(2,2,3)                    # change the number of dimension by reshape
tensor([[[1., 1., 1.],
         [1., 2., 2.]],

        [[2., 2., 3.],
         [3., 3., 3.]]])

print(t.reshape(1,12))
print(t.reshape(1,12).shape)
tensor([[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]])

torch.Size([1, 12])

print(t.view(1,12))
print(t.view(1,12).shape)
tensor([[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]])

torch.Size([1, 12])
```

### Changing shape by squeezing and unsqueezing

> We also can use reshape to squeeze/unsqueeze or flatten a tensor

#### *squeeze and unsqueeze*

The next way we can change the shape of our tensors is by ***squeezing*** and ***unsqueezing*** them.

- *Squeezing* a tensor removes the dimensions or axes that have a length of one.
- *Unsqueezing* a tensor adds a dimension with a length of one.

```python
> print(t.reshape([1,12]))
> print(t.reshape([1,12]).shape)
tensor([[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]])
torch.Size([1, 12])

> print(t.reshape([1,12]).squeeze())
> print(t.reshape([1,12]).squeeze().shape)
tensor([1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.])
torch.Size([12])

> print(t.reshape([1,12]).squeeze().unsqueeze(dim=0))
> print(t.reshape([1,12]).squeeze().unsqueeze(dim=0).shape)
tensor([[1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3.]])
torch.Size([1, 12])
```

#### *Flatten a tensor*

`torch.flatten(input,start_dim=0,end_dim=-1)→ Tensor`

- Flattens a contiguous range of dims in a tensor.

- Parameters

  - **input**([*Tensor*](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")) – the input tensor

  - **start_dim**([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)) – the first dim to flatten

  - **end_dim**([*int*](https://docs.python.org/3/library/functions.html#int "(in Python v3.7)) – the last dim to flatten

#### *Concatenating tensors*

**Combine tensors** using `cat()` function, and the resulting tensor will have a shape that depends on the shape of the two input tensors.

```python
> t1 = torch.tensor([
    [1,2],
    [3,4]
])
> t2 = torch.tensor([
    [5,6],
    [7,8]
])
```

```python
> torch.cat((t1, t2), dim=0)
tensor([[1, 2],
        [3, 4],
        [5, 6],
        [7, 8]])
```

```python
> torch.cat((t1, t2), dim=1)
tensor([[1, 2, 5, 6],
        [3, 4, 7, 8]])
```

When we concatenate tensors, we increase the number of elements contained within the resulting tensor. `dim` specify which dimension we choose to combine the tensors.
