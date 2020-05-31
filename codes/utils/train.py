import time


def train(device, model, train_loader, criterion, optimizer):
    model.train()

    running_losses = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward + backward + step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_losses += loss.item()

    return running_losses


def validate(device, model, val_loader, criterion):
    model.eval()

    running_losses = 0.0
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_losses += loss.item()

    return running_losses


def run(num_epochs, device, model, train_loader, val_loader, criterion, optimizer):
    start = time.time()
    model = model.to(device)

    # make lists to store training and validation results
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    epochs_so_far = 0
    # for each epoch
    for epoch in range(1, num_epochs+1):
        tic = time.time()
        print(f'Epoch {epoch}/{num_epochs}', end=':\t')

        # train
        running_losses = train(device, model, train_loader, criterion, optimizer)
        train_loss = running_losses / len(train_loader)
        train_loss_per_epoch.append(train_loss)

        # validate
        running_losses = validate(device, model, val_loader, criterion)
        val_loss = running_losses / len(val_loader)
        val_loss_per_epoch.append(val_loss)

        toc = time.time()
        epochs_so_far += 1
        print(f'{toc - tic:.2f}s/epoch | Train loss: {train_loss:.4f} | Val. loss: {val_loss:.4f}')

    end = time.time()
    print(f'Training finished in {end - start:.2f}s for {epochs_so_far} epochs...')

    return train_loss_per_epoch, val_loss_per_epoch