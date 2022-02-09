import torch
import torch.nn as nn

class CreatingUserId(nn.Module):
    def __init__(self,
                 num_dayofweek,
                 num_time,
                 num_sex,
                 num_age,
                 num_month,
                 num_day,
                 num_dim=8,
                 num_factor=32,
                 ):
        super(CreatingUserId, self).__init__()

        # userId Embedding
        self.dayofweek_embedding = nn.Embedding(num_embeddings=num_dayofweek,
                                                  embedding_dim=num_dim)
        self.time_embedding = nn.Embedding(num_embeddings=num_time,
                                                  embedding_dim=num_dim)
        self.sex_embedding = nn.Embedding(num_embeddings=num_sex,
                                                  embedding_dim=num_dim)
        self.age_embedding = nn.Embedding(num_embeddings=num_age,
                                                  embedding_dim=num_dim)
        self.month_embedding = nn.Embedding(num_embeddings=num_month,
                                                  embedding_dim=num_dim)
        self.day_embedding = nn.Embedding(num_embeddings=num_day,
                                                  embedding_dim=num_dim)
        self.Embedding_list = nn.ModuleList([self.dayofweek_embedding, self.time_embedding,self.sex_embedding,
                                        self.age_embedding, self.month_embedding, self.day_embedding])

        # self.MLP = nn.Sequential(nn.Linear(num_dim * 6, num_factor),
        #                                   nn.BatchNorm1d(num_factor),
        #                                   nn.LeakyReLU()
        #                                   )
        self.init_weight()

    def init_weight(self):
            # Embedding weight initialization(normal|uniform)
            for layer in self.Embedding_list:
                if isinstance(layer,nn.Embedding):
                    nn.init.normal_(layer.weight)

            # MLP weight initialization
            for layer in self.MLP:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight)

    def forward(self, dayofweek, time, sex, age, month, day):
        # Embedding
        dayofweek_embedded = self.dayofweek_embedding(dayofweek)
        time_embedded = self.time_embedding(time)
        sex_embedded = self.sex_embedding(sex)
        age_embedded = self.age_embedding(age)
        month_embedded = self.month_embedding(month)
        day_embedded = self.day_embedding(day)

        # dayofweek, time, sex, age, month, day embedding concatenation
        user_vector = torch.cat([dayofweek_embedded, time_embedded, sex_embedded, age_embedded, month_embedded, day_embedded], dim=-1)
        # output_userId = self.MLP(user_vector)
        output_userId = user_vector
        return output_userId