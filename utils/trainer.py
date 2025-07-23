import torch
from utils.save import save

class Trainer:

    def __init__(self, model, optimizer=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device if device is not None else next(model.parameters()).device

    def _train(self, dataloader):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.train()
        train_loss, correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = model.loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum()
            # print("loss:", loss.dtype, "; correct:", correct.dtype)

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * dataloader.batch_size # len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            # skip training
            if batch == 5:
                break
        train_loss /= num_batches
        correct /= size
        return train_loss.item(), correct.item()

    def _validate(self, dataloader):
        model = self.model
        optimizer = self.optimizer
        device = self.device
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = torch.tensor(0., device=device), torch.tensor(0., device=device)
        with torch.no_grad(): # temporarily changing global state - torch.is_grad_enabled()
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += model.loss_fn(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum()
                # skip training
                if batch == 5:
                    break
            test_loss /= num_batches
            correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss.item(), correct.item()


    def fit(self, train_dataloader, val_dataloader, epochs, savename="checkpoint"):
        train_losses, train_accuracies = [], []
        val_losses, val_accuracies = [], []
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            # train
            train_loss, train_acc = self._train(train_dataloader)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            # test
            val_loss, val_acc = self._validate(val_dataloader)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            # save
            save(self.model,
                train_losses=train_losses,
                train_accuracies=train_accuracies,
                val_losses=val_losses,
                val_accuracies=val_accuracies,
                savename=savename,
                index=t
            )
        print("Done!")

    def test(model, dataloader, device=None):
        trainer = Trainer(model, device=device)
        return trainer._validate(dataloader)
        # model.eval()
        # x, y = test_data[0]
        # with torch.no_grad():
        #     x = x.unsqueeze(0).to(device) # [N, C, H, W]
        #     pred = model(x)
        #     predicted, actual = dataset.classes[pred[0].argmax(0).item()], dataset.classes[y]
        #     print(f'Predicted: "{predicted}", Actual: "{actual}"')




