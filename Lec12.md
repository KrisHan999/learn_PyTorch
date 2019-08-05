## Tensors for Deep Learning - Broadcasting and Element-wise Operations with PyTroch

```python
> t1 = torch.tensor([
    [1,2],
    [3,4]
], dtype=torch.float32)

> t2 = torch.tensor([
    [9,8],
    [7,6]
], dtype=torch.float32)
```

### Broadcasting

Broadcasting is the concept whose implementation allows us to add scalars to higher dimesional tensors.

### Arithmetic operations are element-wise operations

- Using symblic operations

  ```python
  > print(t + 2)
  tensor([[3., 4.],
          [5., 6.]])
  
  > print(t - 2)
  tensor([[-1.,  0.],
          [ 1.,  2.]])
  
  > print(t * 2)
  tensor([[2., 4.],
          [6., 8.]])
  
  > print(t / 2)
  tensor([[0.5000, 1.0000],
          [1.5000, 2.0000]])
  ```

- Using built-in tensor object methods

  ```python
  > print(t1.add(2))
  tensor([[3., 4.],
          [5., 6.]])
  
  > print(t1.sub(2))
  tensor([[-1.,  0.],
          [ 1.,  2.]])
  
  > print(t1.mul(2))
  tensor([[2., 4.],
          [6., 8.]])
  
  > print(t1.div(2))
  tensor([[0.5000, 1.0000],
          [1.5000, 2.0000]])
  ```

### Comparison operations are element-wise

```python
> t.eq(0)                        # this is equal to t == 0

tensor([[1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]], dtype=torch.uint8)


> t.ge(0)                        # this is equal to t >= 0

tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], dtype=torch.uint8)


> t.gt(0)                        # this is equal to t > 0

tensor([[0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]], dtype=torch.uint8)


> t.lt(0)                        # this is equal to t < 0

tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]], dtype=torch.uint8)

> t.le(7)                        # this is equal to t <= 0

tensor([[1, 1, 1],
        [1, 1, 1],
        [1, 0, 1]], dtype=torch.uint8)
```

### ELement-wise operations using functions

```python
> t.abs() 
tensor([[0., 5., 0.],
        [6., 0., 7.],
        [0., 8., 0.]])


> t.sqrt()
tensor([[0.0000, 2.2361, 0.0000],
        [2.4495, 0.0000, 2.6458],
        [0.0000, 2.8284, 0.0000]])

> t.neg()
tensor([[-0., -5., -0.],
        [-6., -0., -7.],
        [-0., -8., -0.]])

> t.neg().abs()
tensor([[0., 5., 0.],
        [6., 0., 7.],
        [0., 8., 0.]])
```
