import torch

class CategoricalAccuracy(torch.nn.Module):
    def __init__(self):
        super(CategoricalAccuracy, self).__init__()

    def forward(self, y_pred, y_true):
        # TODO (adding keep dimention)
        _, preds = torch.max(y_pred, dim=1)
        correct = (preds == torch.argmax(y_true, dim=1)).float()
        accuracy = correct.sum() / len(correct)
        return accuracy