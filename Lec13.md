## Code for Deep Learning - Argmax and Reduction Tensor Ops

### Tensor reduction operations

> A reduction operation on a tensro is an operation that reduces the number of elements contained within the tensor

```python
> t = torch.tensor([
    [0,1,0],
    [2,0,2],
    [0,3,0]
], dtype=torch.float32)
```

#### *Common tensor reduction operations*

```python
> t.sum()
tensor(8.)

> t.prod()
tensor(0.)

> t.mean()
tensor(.8889)

> t.std()
tensor(1.1667)
```

#### *Reducing tensors by axes*

```python
> t = torch.tensor([
    [1,1,1,1],
    [2,2,2,2],
    [3,3,3,3]
], dtype=torch.float32)
```

```python
> t.sum(dim=0)
tensor([6., 6., 6., 6.])

> t.sum(dim=1)
tensor([ 4.,  8., 12.])
```

#### *Argmax tensor reduction operation*

> *Argmax*returns the index location of the maximum value inside a tensor.

```python
t = torch.tensor([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]
], dtype=torch.float32)
```

```python
> t.max()
tensor(5.)

> t.argmax()
tensor(11)                # 11 -> the index of the max element after flattening


> t.flatten()
tensor([1., 0., 0., 2., 0., 3., 3., 0., 4., 0., 0., 5.])
```

**How to work with specified axes?**

```python
> t.max(dim=0)
(tensor([4., 3., 3., 5.]), tensor([2, 1, 1, 2]))

> t.argmax(dim=0)
tensor([2, 1, 1, 2])

> t.max(dim=1)
(tensor([2., 3., 5.]), tensor([3, 1, 3]))

> t.argmax(dim=1)
tensor([3, 1, 3])
```

> After we specify the axes, max() operation returns two tensors.
> 
> The first tensor contains the max value along the axes.
> 
> The second tendon contains the index locations for the max values.

### Accessing elements inside tensors

```python
> t = torch.tensor([
    [1,2,3],
    [4,5,6],
    [7,8,9]
], dtype=torch.float32)

> t.mean()
tensor(5.)

> t.mean().item()
5.0

> t.mean(dim=0).tolist()
[4.0, 5.0, 6.0]

> t.mean(dim=0).numpy()
array([4., 5., 6.], dtype=float32)
```

- If we want to get value of a single number, we use `item()` tensor method

- If we want to get multiple values:

  - `tolist()`

  - `numpy()`
