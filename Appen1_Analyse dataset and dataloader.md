

```python
import torch
import torchvision.transforms as transforms
import torchvision
```

# Dataset


```python
data_set = torchvision.datasets.FashionMNIST(
    root='./data',
    download=True,
    train=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
```

## Len


```python
print(len(data_set))
```

    60000
    

## Access sample


```python
image, label = next(iter(data_set))
print(image.shape)
print(label)
```

    torch.Size([1, 28, 28])
    9
    

## Access all data


```python
print(data_set.data.shape)
print(data_set.targets.shape)
```

    torch.Size([60000, 28, 28])
    torch.Size([60000])
    

# Dataloader


```python
data_loader = torch.utils.data.DataLoader(
    data_set,
    batch_size=100,
    shuffle=True
)
```

## Len


```python
len(data_loader)
```




    600



## Access batch


```python
batch = next(iter(data_loader))
images, labels = batch
print(images.shape)
print(labels.shape)
```

    torch.Size([100, 1, 28, 28])
    torch.Size([100])
    

## Access all data


```python
print(data_loader.dataset.data.shape)
print(data_loader.dataset.targets.shape)
```

    torch.Size([60000, 28, 28])
    torch.Size([60000])
    


```python

```
