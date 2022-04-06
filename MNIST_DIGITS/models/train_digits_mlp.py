from nn_lib.mdl import CELoss
from nn_lib.optim import SGD
from nn_lib.data import Dataloader

from toy_mlp.model_trainer import ModelTrainer

from MNIST_DIGITS.models.mlp_classifier import MLPClassifier
from MNIST_DIGITS.models.digits_dataset import DigitsMNISTDataset


def main(n_epochs, hidden_layer_sizes):
    # create MLP classification model
    mlp_model = MLPClassifier(in_features=8*8, number_of_classes=10, hidden_layer_sizes=hidden_layer_sizes)
    print(f'Created the following MLP classifier:\n{mlp_model}')
    # create loss function
    loss_fn = CELoss()
    # create optimizer for model parameters
    optimizer = SGD(mlp_model.parameters(), lr=1e-2, weight_decay=5e-4)

    # create a model trainer
    model_trainer = ModelTrainer(mlp_model, loss_fn, optimizer)

    # generate a training dataset
    train_dataset = DigitsMNISTDataset(train=True)
    # generate a validation dataset different from the training dataset
    val_dataset = DigitsMNISTDataset(train=False)
    # create a dataloader for training data with shuffling and dropping last batch
    train_dataloader = Dataloader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
    # create a dataloader for validation dataset without shuffling or last batch dropping
    val_dataloader = Dataloader(val_dataset, batch_size=100, shuffle=False, drop_last=False)

    # train the model for a given number of epochs
    model_trainer.train(train_dataloader, n_epochs=n_epochs)

    # validate model on the train data
    # Note: we create a new dataloader without shuffling or last batch dropping
    train_predictions, train_accuracy, train_mean_loss = model_trainer.validate(
        Dataloader(train_dataset, batch_size=100, shuffle=False, drop_last=False))
    print(f'Train accuracy: {train_accuracy:.4f}')
    print(f'Train loss: {train_mean_loss:.4f}')

    # validate model on the validation data
    val_predictions, val_accuracy, val_mean_loss = model_trainer.validate(val_dataloader)
    print(f'Validation accuracy: {val_accuracy:.4f}')
    print(f'Validation loss: {val_mean_loss:.4f}')

    val_dataset.vizualize_for_classifier(val_predictions)


if __name__ == '__main__':
    main(n_epochs=100, hidden_layer_sizes=(256, ))
