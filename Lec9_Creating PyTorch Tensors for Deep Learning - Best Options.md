## Creating PyTorch Tensors for Deep Leaerning - Best Options

As we already know, there are four ways to create a `torch.Tensor` object.

- `torch.Tensor(data)`

- `torch.tensor(data)`

- `torch.as_tensor(data)`

- `torch.from_numpy(data)`

```python
> data = np.array([1,2,3])
> type(data)
numpy.ndarray

> o1 = torch.Tensor(data)
> o2 = torch.tensor(data)
> o3 = torch.as_tensor(data)
> o4 = torch.from_numpy(data)

> print(o1)
> print(o2)
> print(o3)
> print(o4)
tensor([1., 2., 3.])
tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3], dtype=torch.int32)
tensor([1, 2, 3], dtype=torch.int32)
```

### Tensor creation operations: *What's the difference?*

#### *Uppercase/lowercase: `torch.Tensor()` vs `torch.tensor()`*

`torch.Tensor()` is the constructor of the `torch.Tensor` class,  `torch.tensor()` is what we call a *factory function* that constructs `torch.Tensor` object and returns them to the caller, other two functions, `torch.as_tensor()` and `torch.from_numpy()` are also factory functions.

**Difference:**

However, the **factory function** `torch.tensor()` has **better documentation and more configuration options**.

#### *default `dtype` vs inferred `dtype`*

```python
> print(o1.dtype)
> print(o2.dtype)
> print(o3.dtype)
> print(o4.dtype)
torch.float32
torch.int32
torch.int32
torch.int32

> torch.get_default_dtype()
torch.float32

> o1.dtype == torch.get_default_dtype()
True
```

`torch.Tensor()` constructor uses the **default** `dtype` when building the tensor. The other calls choose a dtype based on the incoming data. This is called ***type inference***. 

Note that the `dtype` can also be explicitly set for these calls bt specifying the `dtype` as an argument. While `torch.Tensor()` can't.

```python
> torch.tensor(data, dtype=torch.float32)
> torch.as_tensor(data, dtype=torch.float32)
```

#### *Sharing memory for performance: copy vs share*

```python
> print('old:', data)
old: [1 2 3]

> data[0] = 0

> print('new:', data)
new: [0 2 3]

> print(o1)
> print(o2)
> print(o3)
> print(o4)

tensor([1., 2., 3.])
tensor([1, 2, 3], dtype=torch.int32)
tensor([0, 2, 3], dtype=torch.int32)
tensor([0, 2, 3], dtype=torch.int32)
```

| Share Data         | Copy Data      |
| ------------------ | -------------- |
| torch.as_tensor()  | torch.tensor() |
| torch.from_numpy() | torch.Tensor() |

If we have a `torch.Tensor` and we wat to convert it to a `numpy.ndarray`, we do it like:

```python
> n3 = o3.numpy()
> print(o3.numpy())
[0 2 3]

> n3[0] = 1
> print(o3)
> tensor([1, 2, 3], dtype=torch.int32)
```

As we can see,  returned numpu array from`x.numpy()` share memory with tensor object.

**Difference**

The `torch.from_numpy()` function only accepts `numpy.ndarray` s, while the `torch.as_tensor()` function accepts a wide variety of Python array-like objects including other PyTorch tensors.



### Best options for creating tensors in PyTorch

Given all of these details, these two are the best options:

- `torch.tensor()`
- `torch.as_tensor()`

**Some things to keep in mind(memory sharing works where it can)**

- Since `numpy.ndarray` objects are allocated on the CPU, the `as_tensor()` function must copy the data from the CPU to the GPU when a GPU is being used.

- The memory sharing of `as_tensor()` doesn’t work with built-in Python data structures like lists.

  ```python
  > l = [1,2,3,4]
  
  > t = torch.as_tensor(l)
  
  > l[1] = 0
  
  > print(t)
  tensor([1, 2, 3, 4])
  # as_tensor() doesn't share memory with built-in Python data structure
  ```

- The `as_tensor()` call requires developer knowledge of the sharing feature.

- The `as_tensor()` performance improvement will be greater when there are a lot of back and forth operations between `numpy.ndarray` objects and tensor objects.
