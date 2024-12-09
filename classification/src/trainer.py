import time, torch

from src.utils.statusbar import print_status_bar
from src.utils.tools import average
from src.validate import validate
from src.models.CategoricalAccuracy import CategoricalAccuracy

from src.utils import trace, modelfile, tools

def train_model(parameters, model, device, train_loader, valid_loader, train_dataset_size):

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters.initial_lr)
    criterion = torch.nn.CrossEntropyLoss()
    categorical_accuracy = CategoricalAccuracy()

    batches = train_dataset_size//parameters.batch
    start_epoch, min_loss, metrics_train, metrics_valid = modelfile.init_metrics(parameters=parameters)  

    tools.print_separator()
    for epoch in range(start_epoch, parameters.epochs):
        model.train()
        loss_list, accuracy_list= [], []

        start_time = time.time()
        for i, data in enumerate(train_loader):
            x, target = data
            x = x.to(device)
            target = target.to(device)

            # inference      
            output = model(x)

            # accuracy
            accuracy = categorical_accuracy(output, target)
            accuracy_list.append(accuracy.item())

            # loss
            loss = criterion(output, target)
            loss_list.append(loss.item())
            loss.backward()

            # weights update 
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            print_status_bar( i + 1, batches, loss=[loss.item()], elapsed_time = time.time() - start_time, epoch=epoch, epochs=parameters.epochs)

        # train (metrics)
        metrics_train['loss'].append( average(loss_list) )
        metrics_train['accuracy'].append( average(accuracy_list) )

        # validate (metrics)
        metrics_valid = validate(device=device, model=model, metrics=metrics_valid, valid_loader=valid_loader, epochs=parameters.epochs)

        # save trace metrics
        trace.trace(parameters=parameters, epoch=epoch, metrics_train=metrics_train, metrics_valid=metrics_valid )

        # save latest and best model
        min_loss = modelfile.save_model(parameters=parameters, epoch=epoch, min_loss=min_loss, metrics_train=metrics_train, metrics_valid=metrics_valid, model=model)

    return model
