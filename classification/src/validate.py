import time
import torch

from src.utils.tools import average
from src.utils.statusbar import print_status_bar
from src.models.CategoricalAccuracy import CategoricalAccuracy

def validate(device, model, metrics, valid_loader, epochs):
    loss_list, accuracy_list = [], []
    criterion = torch.nn.CrossEntropyLoss()
    categorical_accuracy = CategoricalAccuracy()
    with torch.no_grad():
        model.eval()
        start_time = time.time()
        for j, data in enumerate(valid_loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)
            output = model(x)

            # loss
            loss = criterion(output, target)
            loss_list.append(loss.item())

            # accuracy
            accuracy = categorical_accuracy(output, target)
            accuracy_list.append(accuracy.item())
            tmp_average = [average(loss_list)]
            print_status_bar(j+1, len(valid_loader), tmp_average, elapsed_time=time.time() - start_time, epoch=None, epochs=epochs)

        metrics['loss'].append( average(loss_list) )
        metrics['accuracy'].append( average(accuracy_list) )
        return metrics
