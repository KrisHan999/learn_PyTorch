## PyTorch Tensors Explained - Neural Network Programming

#### *Tensor attributes*

Every `torch.Tensor`  has these attributes:

- `torch.dtype`

- `torch.device`

- `torch.layout`

#### *Tensors have a `torch.dtype`*

| *D*ata type              | dtype         | CPU tensor         | GPU tensor              |
| ------------------------ | ------------- | ------------------ | ----------------------- |
| 32-bit floating point    | torch.float32 | torch.FloatTensor  | torch.cuda.FloatTensor  |
| 64-bit floating point    | torch.float64 | torch.DoubleTensor | torch.cuda.DoubleTensor |
| 16-bit floating point    | torch.float16 | torch.HalfTensor   | torch.cuda.HalfTensor   |
| 8-bit integer (unsigned) | torch.uint8   | torch.ByteTensor   | torch.cuda.ByteTensor   |
| 8-bit integer (signed)   | torch.int8    | torch.CharTensor   | torch.cuda.CharTensor   |
| 16-bit integer (signed)  | torch.int16   | torch.ShortTensor  | torch.cuda.ShortTensor  |
| 32-bit integer (signed)  | torch.int32   | torch.IntTensor    | torch.cuda.IntTensor    |
| 64-bit integer (signed)  | torch.int64   | torch.LongTensor   | torch.cuda.LongTensor   |

One thing to keep in mind about tensor data types is that **tensor operations between tensors must happen between tensors with the same types of data.**

#### *Tensors have a `tensor.device`*

The device specifiesthe **device** (CPU or GPU) where the **tensor's data is allocated**. This determines **where tensor computations for the given tensor will be performed**.

```python
> device = torch.device('cuda:0')
> device
device(type='cuda', index=0)
```

#### *Tensors have a `torch.layout`*

### Creating tensors using data

These are the primary ways of creating tensor objects with data ()array-like) in PyTorch:

- `torch.Tensor(data)`

- `torch.tensor(data)`

- `torch.as_tensor(data)`

- `torch.from_numpy(data)`

```python
> data = np.array([1,2,3])
> type(data)
numpy.ndarray
```

```python
> o1 = torch.Tensor(data)
> o2 = torch.tensor(data)
> o3 = torch.as_tensor(data)
> o4 = torch.from_numpy(data)

> print(o1)
> print(o2)
> print(o3)
> print(o4)
tensor([1., 2., 3.])                    # torch.Tensor(data) -> float data type

tensor([1, 2, 3], dtype=torch.int32)    # Others -> kepp the original data type

tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3], dtype=torch.int32)
```

All of the options (`o1`,`o2`,`o3`,`o4`) appear to have produced the same tensors except for the first one. The first option (`o1`) has dots after the number indicating that the numbers are`float`s, while the next three options have a type of`int32`.

### Creation options without data

```python
> print(torch.eye(2))
tensor([
    [1., 0.],
    [0., 1.]
])
```

```python
> print(torch.zeros([2,2]))
tensor([
    [0., 0.],
    [0., 0.]
])
```

```python
> print(torch.ones([2,2]))
tensor([
    [1., 1.],
    [1., 1.]
])
```

```python
> print(torch.rand([2,2]))
tensor([
    [0.0465, 0.4557],
    [0.6596, 0.0941]
])
```
