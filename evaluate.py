'''
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''
import torch
import torch.nn as nn

def test( model:nn.Module,
          dataloader:torch.utils.data.dataloader,
          criterion,
          device:torch.device):

    total_batch_len = len(dataloader)
    total_loss = 0
    with torch.no_grad():
        for destination, time, sex, age, dayofweek, month, day, visitor in dataloader:
            destination = destination.to(device)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device), age.to(
                device), month.to(device), day.to(device)
            target = visitor.to(device)
            pred = model(dayofweek, time, sex, age, month, day, destination)
            loss = criterion(pred, target)
            total_loss += loss
    return total_loss/total_batch_len


