import torch
import torch.nn as nn
from UserMLP import CreatingUserId

class MatrixFactorization(nn.Module):
    def __init__(self,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_month,
                 num_day,
                 num_destination,
                 num_dim=8,
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

    def forward(self, users, items):
        output = torch.mm(self.user_embedding(users), torch.transpose(self.item_embedding(items),1,2))
        return output