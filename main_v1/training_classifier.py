import torch
import torch.optim as optim
import numpy as np


def train_network(model,train_loader,val_loader):
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define a loss function (cross-entropy loss)
    criterion = torch.nn.CrossEntropyLoss()

    for e in range(40):
        # Training model
        model.train()
        # y is label, x is feature vector
        for i, (x, y) in enumerate(train_loader):
            #############################################
            # TODO: YOUR CODE HERE
            #############################################
            # print(x.shape)  # Add this line to debug
            # print(y.shape)  # Add this line to debug
            x, y = x.cuda().float(), y.cuda().long()
            # calculate the gradient and update the model
            # Forward pass
            pred = model(x)

            # Calculate the loss
            loss = criterion(pred, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Validating model
        model.eval()
        gt = []
        preds = []
        current_loss = 0
        for i, (x, y) in enumerate(val_loader):
            #############################################
            # TODO: YOUR CODE HERE
            #############################################
            # Calculate the validation outputs and validation loss in current epoch
            x, y = x.cuda().float(), y.cuda().long()
            outputs = model(x)
            current_loss = criterion(outputs,y)

            # Save the predictions for validation sets and ground truth
            preds.append(torch.argmax(outputs, dim=-1).cpu().detach().numpy())
            gt.append(y.cpu().numpy())

        # Change prediction and ground truth to numpy
        preds = np.concatenate(preds, axis=0)
        gt = np.concatenate(gt, axis=0)
        # Calculate difference and mae
        diff = preds - gt
        mae = np.abs(diff).mean()
        print(f"Epoch: {e} \t mae: {mae:.3f} \t loss: {current_loss:.3f}")

    print("=> training finished")
    return model, preds, gt