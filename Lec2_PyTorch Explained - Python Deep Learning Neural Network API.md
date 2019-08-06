## Pytorch Explained - Python Deep Learning Neural Network API



### PyTorch -> Numpy and GPU

PyTorch is **a deep learning framework** and **a scientific computing package**. **The scientific computing aspect** of Pytorch is primarily **a result PyTorch's tensor library and associated tensor operations.**

For example, **PyTorch** `torch.Tensor` objects that are created from **NumPy** `ndarray` objects, share memory. **This makes the transition between PyTorch and NumPy very cheap from a preformance perspective.**



With PyTorch tensors, **GPU support is built-in**. It's very easy with PyTorch to move tensors to and from a GPU if we have one installed on our system.



### PyTorch Package

| Package             | Description                                                                                                                                                      |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| torch               | The top-level PyTorch package and tensor library.                                                                                                                |
| torch.nn            | A subpackage that contains modules and extensible classes for building neural networks.                                                                          |
| torch.autograd      | A subpackage that supports all the differentiable Tensor operations in PyTorch.                                                                                  |
| torch.nn.functional | A functional interface that contains typical operations used for building neural networks like loss functions, activation functions, and convolution operations. |
| torch.optim         | A subpackage that contains standard optimization operations like SGD and Adam.                                                                                   |
| torch.utils         | A subpackage that contains utility classes like data sets and data loaders that make data preprocessing easier.                                                  |
| torchvision         | A package that provides access to popular datasets, model architectures, and image transformations for computer vision.                                          |



### PyTorch -> Dynamic computational graph

PyTorch uses a computational graph that is called a dynamic computational graph. This means that the graph is generated on the fly as the operations are created.
















