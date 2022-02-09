'''
made by changhyeonnam
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''
import torch
import torch.nn as nn
from model.UserMLP import CreatingUserId

class MatrixFactorization(nn.Module):
    def __init__(self,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_month,
                 num_day,
                 num_destination,
                 num_dim=4,
                 num_factor=32,
                 ):
        super(MatrixFactorization,self).__init__()

        self.user_embedding = CreatingUserId(num_dayofweek= num_dayofweek,
                                             num_time = num_time,
                                             num_sex = num_sex,
                                             num_age = num_age,
                                             num_month = num_month,
                                             num_day = num_day,
                                             num_dim=num_dim,
                                             num_factor=num_factor)
        self.item_embedding = nn.Embedding(num_destination, num_factor)
        nn.init.normal_(self.item_embedding.weight)

    def forward(self,  dayofweek, time, sex, age, month, day, destination):
        user_embedding = self.user_embedding(dayofweek, time, sex, age, month, day)
        item_embedding = self.item_embedding(destination)
        # print(f'user_embedding: {user_embedding.shape}, item_embedding: {item_embedding.shape}')

        output = torch.mm(user_embedding,
                          torch.transpose(item_embedding,0,1))

        output = torch.diagonal(output,0)
        output = output.view(-1)
        # print(f'output shpae: {output.shape}')
        return output
