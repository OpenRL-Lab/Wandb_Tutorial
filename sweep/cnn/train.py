import os

import wandb

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

from fashion_data import fashion

from model import CNNModel
from config import hyperparameter_defaults

def train():
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = fashion(root='./data',
                                train=True,
                                transform=transform,
                                download=True
                               )

    test_dataset = fashion(root='./data',
                                train=False,
                                transform=transform,
                               )

    label_names = [
        "T-shirt or top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Boot"]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config.batch_size,
                                              shuffle=False)


    model = CNNModel(config)
    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    iter = 0
    for epoch in range(config.epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = Variable(images)
            labels = Variable(labels)

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(images)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                # Calculate Accuracy
                correct = 0.0
                correct_arr = [0.0] * 10
                total = 0.0
                total_arr = [0.0] * 10

                # Iterate through test dataset
                for images, labels in test_loader:
                    images = Variable(images)

                    # Forward pass only to get logits/output
                    outputs = model(images)

                    # Get predictions from the maximum value
                    _, predicted = torch.max(outputs.data, 1)

                    # Total number of labels
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                    for label in range(10):
                        correct_arr[label] += (((predicted == labels) & (labels==label)).sum())
                        total_arr[label] += (labels == label).sum()

                accuracy = correct / total

                metrics = {'accuracy': accuracy, 'loss': loss}
                for label in range(10):
                    metrics['Accuracy ' + label_names[label]] = correct_arr[label] / total_arr[label]

                wandb.log(metrics)

                # Print Loss
                print('Iteration: {0} Loss: {1:.2f} Accuracy: {2:.2f}'.format(iter, loss, accuracy))
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))