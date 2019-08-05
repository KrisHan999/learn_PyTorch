## CNN Training with Code Example - Calculate Loss, Gradient & Update Weights

During the entire training process, we do as many epochs as necessary to reach our desired level of accuracy. With this, we have the following steps:

1. Get batch from the training set.

2. Pass batch to network.

3. Calculate the loss (difference between the predicted values and the true values).

4. Calculate the gradient of the loss function w.r.t the network's weights.

5. Update the weights using the gradients to reduce the loss.

6. Repeat steps 1-5 until one epoch is completed.

7. Repeat steps 1-6 for as many epochs required to obtain the desired level of accuracy.



### The Training Process

```python
> torch.set_grad_enabled(True)        # by default is True

<torch.autograd.grad_mode.set_grad_enabled at 0x15b22d012b0>
```

#### *Preparing for the Forward Pass*

We'll begin by:

1. Creating an instance of our`Network`class.
2. Creating a data loader that provides batches of size`100`from our training set.
3. Unpacking the images and labels from one of these batches.

```python
> network = Network()

> train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
> batch = next(iter(train_loader)) # Getting a batch
> images, labels = batch
```



#### *Calculating the loss*

```python
> preds = network(images)
> loss = F.cross_entropy(preds, labels) # Calculating the loss

> loss.item()
2.307542085647583

> get_num_correct(preds, labels)
9
```



#### *Calculating the Gradients*

```python
loss.backward() # Calculating the gradients
```



#### *Updating the Weights*

```python
optimizer = optim.Adam(network.parameters(), lr=0.01)
optimizer.step() # Updating the weights
```



### Train Using a Single Batch

```python
network = Network()

train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
optimizer = optim.Adam(network.parameters(), lr=0.01)

batch = next(iter(train_loader)) # Get Batch
images, labels = batch

preds = network(images) # Pass Batch
loss = F.cross_entropy(preds, labels) # Calculate Loss

# First, calculate the gradient, then update weights
loss.backward() # Calculate Gradients
optimizer.step() # Update Weights


print('loss1:', loss.item())
preds = network(images)
loss = F.cross_entropy(preds, labels)
print('loss2:', loss.item())
```



### Entire Training Process

```python

# The training loop
for epoch in range(10):
    total_correct = 0
    total_loss = 0
    for batch in train_loader:
        images, labels = batch
        
        # Every time a variable is back propogated through, 
        # the gradient will be accumulated instead of being replaced.
        # So, we need to zero the grad of previous training
        optimizer.zero_grad()

        preds = network(images)
        
        loss = F.cross_entropy(preds, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()

    print('epoch:', epoch, "total_correct:", total_correct, "loss:", total_loss)
```






