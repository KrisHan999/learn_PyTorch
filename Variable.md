# “PyTorch - Variables, functionals and Autograd.”

[https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/](https://jhui.github.io/2018/02/09/PyTorch-Variables-functionals-and-Autograd/)



### Variables

A **Variable** wraps a Tensor. It supports nearly all the API’s defined by a Tensor. Variable also **provides a *backward* method to perform backpropagation**.



```python
>import torch

>from torch.autograd import Variable

# Set the requires_grad to True, otherwise it can't get the gradient
>x = Variable(torch.ones(2,2), requires_grad=True)
>y = x+2
>z = y*y*2

>out = z.mean()

# out is the outcome of the x, and since X is the Variable type, could use out.backward() to get the gradient of x
>out.backward()
>print(x.grad)

# X is Variable type, it has gradient
tensor([[3., 3.],
        [3., 3.]])
>print(y.grad)
None
>print(z.grad)
None
```


