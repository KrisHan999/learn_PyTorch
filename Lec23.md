## Neural Networks Batch Processing - Pass Image Batch to PyTorch CNN

```python
> data_loader = torch.utils.data.DataLoader(
     train_set, batch_size=10
)

> batch = next(iter(data_loader))
> images, labels = batch

> images.shape
torch.Size([10, 1, 28, 28])

> labels.shape
torch.Size([10])

> preds = network(images)

> preds.shape            # (batch size, number of prediction classes)

torch.Size([10, 10])
```


