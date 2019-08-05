## PyTorch Datasets and DataLoaders - Training Set Exploration for Deep Learning and AI

### PyTorch Dataset: Working with the training set

We already get the train_set from previous lecture.

```python
> len(train_set)
60000

> train_set.targets
tensor([9, 0, 0, ..., 3, 0, 5])
```

#### *Accessing data in the training set*

```python
> sample = next(iter(train_set))
> len(sample)
2
> image, label = sample
> type(image)
torch.Tensor

> type(label)
int

> image.shape
torch.Size([1, 28, 28]) 

> torch.tensor(label).shape
torch.Size([])
```

### PyTorch DataLoader: Working with batches of data

```python
> display_loader = torch.utils.data.DataLoader(
    train_set, batch_size=10
)

# note that each batch will be different when shuffle=True
> batch = next(iter(display_loader))
> print('len:', len(batch))
len: 2

> images, labels = batch

> print('types:', type(images), type(labels))
> print('shapes:', images.shape, labels.shape)
types: <class 'torch.Tensor'> <class 'torch.Tensor'>
shapes: torch.Size([10, 1, 28, 28]) torch.Size([10])
```



#### *Plot a batch of images*

`torchvision.utils.make_grid()`

```python
> grid = torchvision.utils.make_grid(images, nrow=10)        # nrow -> number of image in a row


> plt.figure(figsize=(15,15))
> plt.imshow(np.transpose(grid, (1,2,0)))

> print('labels:', labels)
labels: tensor([9, 0, 0, 3, 0, 2, 7, 2, 5, 5])
```
