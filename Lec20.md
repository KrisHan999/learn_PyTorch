## Callable Neural Networks - Linear Layer in Depth

```python
in_features = torch.tensor([1,2,3,4], dtype=torch.float32)
fc = nn.Linear(in_features=4, out_features=3, bias=False)
> fc(in_features)
tensor([-0.8877,  1.4250,  0.8370], grad_fn=<SqueezeBackward3>)
```

We can **call the object instance** like this because PyTorch neural network modules are *callable Python objects*.

**`__call__()`function**

What makes this possible is that PyTorch module classes implement another special Python Function called `__call__()`. **If a class implement the `__call__()` method, the special call method will be invoked anytime the object instance is called.**

#### *Instead of calling forawrd(), call the object instance directly!*

Instead of calling the `forward()` method directly, we call the object instance, after the object instance is called, the `__call__()` method is invoked under the hood, and the `__call__()` in turn invokes the `forward()` method.


