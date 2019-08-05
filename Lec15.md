## CNN Image Preparation Code Project - Learn to Extract, Transform, Load (ETL)

#### *The ETL process*

- Extract data from a data structure

- Transform data into a desirable format

- Load data into a suitable structure

#### *PyTorch imports*

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
```

| Package                | Description                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| torch                  | The top-level PyTorch package and tensor library.                                                                               |
| torch.nn               | A subpackage that contains modules and extensible classes for building neural networks.                                         |
| torch.optim            | A subpackage that contains standard optimization operations like SGD and Adam.                                                  |
| torch.nn.functional    | A functional interface that contains typical operations used for building neural networks like loss functions and convolutions. |
| torchvision            | A package that provides access to popular datasets, model architectures, and image transformations for computer vision.         |
| torchvision.transforms | An interface that contains common transforms for image processing.                                                              |

#### *Other imports*

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
#from plotcm import plot_confusion_matrix

import pdb                                    # Python debugger


torch.set_printoptions(linewidth=120)
```

### Preparing our data using PyTorch

In order to ETL, PyTirch provides us with two classes:

| Class                       | Description                                                 |
| --------------------------- | ----------------------------------------------------------- |
| torch.utils.data.Dataset    | An abstract class for representing a dataset.               |
| torch.utils.data.DataLoader | Wraps a dataset and provides access to the underlying data. |

If we want to **build a new dataset**, `torch.utils.data.Dataset` has method that we must implement. We extend the `Dataset` class by creating a subclass that implements these required methods.

> All subclasses of the Dataset class must override `__len__`, that provides the size of the dataset, and `__getitem__`, supporting integer indexing in range from `0` to `len(self)` exclusive.

### PyTorch torchvision package

The `torchvision` package gives us access to the following resources:

- Datasets

- Models

- Transforms

- Utils

#### *PyTorch Dataset class*

```python
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
```

| Parameter | Description                                                                        |
| --------- | ---------------------------------------------------------------------------------- |
| root      | The location on disk where the data is located.                                    |
| train     | If the dataset is the training set                                                 |
| download  | If the data should be downloaded.                                                  |
| transform | A composition of transformations that should be performed on the dataset elements. |

#### *PyTorch DataLoader class*

```python
train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=1000
    ,shuffle=True
)
```

#### *View the data*

```python
# images -> (B, C, H, W)
# grid -> (C, H, W)
grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))
```


