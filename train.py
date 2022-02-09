'''
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''
import torch
import torch.nn as nn
from evaluate import test

def train(model:nn.Module,
          optimizer:torch.optim,
          epochs:int,
          dataloader:torch.utils.data.dataloader,
          test_dataloader,
          criterion,
          device:torch.device,
          print_cost=True):

    total_batch_len = len(dataloader)

    for epochs in range(0,epochs):
        total_loss = 0
        for destination, time, sex, age, dayofweek, month, day, target in dataloader:
            destination = destination.to(device)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device), age.to(
                device), month.to(device), day.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(dayofweek, time, sex, age, month, day, destination)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss
        train_average_loss = total_loss/total_batch_len

        model.eval()
        test_average_loss = test(model= model,
                                 dataloader= test_dataloader,
                                 criterion= criterion,
                                 device= device,)

        if print_cost :
            print(f"Epoch: {epochs+1} \t Train average RMSE Loss: {train_average_loss} \t Test average RMSE Loss: {test_average_loss}")


