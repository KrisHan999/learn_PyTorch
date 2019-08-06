## CUDA Explained - Why Deep Learning uses GPUs

Nvidia hardware is the hardware.

CUDA is a platform.

PyTorch is architecture that could use the CUDA platform.

## Using CUDA with PyTorch

The tensor object created in this way is on the **CPU by default**.

```python
> t = torch.tensor([1,2,3])
> t
tensor([1, 2, 3])
```

Now, move the tensor onto the **GPU**

```python
> t = t.cuda()
> t
tensor([1, 2, 3], device='cuda:0')
```

Then, move the data back to CPU

```python
> t = t.cpu()

> t
tensor([1, 2, 3])
```

### GPU can be slower than CPU

GPU is only faster for particular tasks. One issue that we can run into is bottlenecks that slows our perfromance. For example, **moving data from CPU to the GPU is costly**, so in this case, **overall performance might be slower if the computation task is a simple one**.

Remember, the GPU works well for task that can be broken into many smaller tasks, and if a computer is already small, we won't have much to gain by moving the task to GPU.
