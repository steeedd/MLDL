from models.customnet import CustomNet

# Train the Model


def train(epoch, model, train_loader, criterion, optimizer):
    # Set the Model on the Training Mode
    model.train()

    # running_loss -> Total Loss across the Whole Epoch
    running_loss = 0.0

    # correct -> #Images Predicted CORRECTLY
    correct = 0

    # total -> #Images Seen
    total = 0

    # Loops over EACH Batch in Training Set
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move inputs (batch of Images) and targets (True Class Labels) to GPU
        inputs, targets = inputs.cuda(), targets.cuda()

        # Pass the inputs (batch of Iamges) to the Model:
        # outputs -> Tensor of logits [B, 200]
        outputs = model(inputs)

        # Compute the Loss/Error
        # between the Predictions (outputs) and the True Labels(targets)
        # loss -> Single Scalar Loss Value
        loss = criterion(outputs, targets)

        # Backpropagation:
        optimizer.zero_grad() # -> Reset the Gradients of Model Parameters
        loss.backward() # -> Compute the Gradients of Loss with respect to EACH Weight & EACH Bias
        optimizer.step() # -> Update the Model's Parameters using the Gradients

        # Add the Scalar Loss Value (.item() is Needed) to the Total Loss
        running_loss += loss.item()

        # Get the Predicted Class (Class with the Max Score from outputs):
        # _, -> NO Save the Max Score
        # predicted -> Save ONLY the Predicted Class associated wit the Max Score
        _, predicted = outputs.max(1)

        # #Samples that were in this Batch:
        # - targets -> Tensor of shape: [B]
        # - .size(0) -> Return #Elements in targets
        total += targets.size(0)

        # Compare Predicted Labels (predicted) and True Labels (True Labels)
        # - .sum() -> Sum the 1s (Correct Predictions in the Batch)
        # - .item() -> Extract the Scalar Value
        correct += predicted.eq(targets).sum().item()

    # Compute the Average Loss per Batch
    train_loss = running_loss / len(train_loader)

    # Compute the Overall Accuracy for the Epoch
    train_accuracy = 100. * correct / total

    # DEBUG:
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')


    # Validate the Model
def validate(model, val_loader, criterion):
    # Set the Model on the Evaluation Mode
    model.eval()

    # val_loos -> Total Loss across ALL Validation Batches
    val_loss = 0

    # correct -> #Images Predicted CORRECTLY
    correct = 0

    # total -> #Images Seen
    total = 0

    # Disable Gradient Calculations (BECAUSE we're in Validation)
    with torch.no_grad():
        # Loops over EACH Batch in the Validation Set
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            # Move inputs (batch of Images) and targets (True Class Labels) to GPU
            inputs, targets = inputs.cuda(), targets.cuda()

            # Pass the inputs (batch of Iamges) to the Model:
            # outputs -> Tensor of logits [B, 200]
            outputs = model(inputs)

            # Compute the Loss/Error
            # between the Predictions (outputs) and the True Labels(targets)
            # loss -> Single Scalar Loss Value
            loss = criterion(outputs, targets)

            # Add the Scalar Loss Value (.item() is Needed) to the Total Loss
            val_loss += loss.item()

            # Get the Predicted Class (Class with the Max Score from outputs):
            # _, -> NO Save the Max Score
            # predicted -> Save ONLY the Predicted Class associated wit the Max Score
            _, predicted = outputs.max(1)

            # #Samples that were in this Batch:
            # - targets -> Tensor of shape: [B]
            # - .size(0) -> Return #Elements in targets
            total += targets.size(0)

            # Compare Predicted Labels (predicted) and True Labels (True Labels)
            # - .sum() -> Sum the 1s (Correct Predictions in the Batch)
            # - .item() -> Extract the Scalar Value
            correct += predicted.eq(targets).sum().item()

    # Compute the Average Loss per Batch
    val_loss = val_loss / len(val_loader)

    # Compute the Overall Accuracy for the Epoch
    val_accuracy = 100. * correct / total

    # DEBUG:
    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy