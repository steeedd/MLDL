from models.customnet import CustomNet
from data.dataloader import getdata
import torch
from torch import nn
import wandb  # Import wandb

# Initialize wandb
wandb.init(project="Lab3", name="experiment_1", config={
    "learning_rate": 0.001,
    "momentum": 0.9,
    "epochs": 10
})

# Train the Model
train_loader, val_loader = getdata()

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total

    # Log to wandb
    wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Epoch": epoch})

    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

def validate(model, val_loader, criterion, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    # Log to wandb
    wandb.log({"Validation Loss": val_loss, "Validation Accuracy": val_accuracy, "Epoch": epoch})

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

# Full Implementation
model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best_acc = 0
num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)
    val_accuracy = validate(model, val_loader, criterion, epoch)
    best_acc = max(best_acc, val_accuracy)

print(f'Best Validation Accuracy: {best_acc:.2f}%')

# Finish wandb run
wandb.finish()
