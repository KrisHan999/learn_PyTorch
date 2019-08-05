## Rank, Axes, and Shape Explained - Tensors for Deep Learning

Rank:

- tell us how many indexes are needed to refer to a specific element within the tensor

Axes:

- an axis of a tensor is a specific dimension of a tensor

Shape:

- the shape of a tensor gives us the length of each axis of the tensor

  ```python
  > dd = [

  > [1,2,3],

  > [4,5,6],

  > [7,8,9]

  > ]
  
  > t = torch.tensor(dd)
  
  > t.reshape(1,9)
  tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
  
  > t.reshape(1,9).shape
  torch.Size([1, 9])
  
  
  # twp ways to get the shape/size of the tensor
  > t.shape
  torch.Size([3,3])
  
  > t.size()
  torch.Size([3,3])
  ```

  



**Reshaping a tensor**

```python
> t.reshape(1,9)
tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])

> t.reshape(1,9).shape
torch.Size([1, 9])
```



Reshaping changes the shape but not the underlying data elements
