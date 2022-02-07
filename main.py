import torch
from parser import parser
import numpy
import pandas
import datetime
from model.MF import MatrixFactorization
from criterion import RMSELoss

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if device == 'cuda':
    print(f'Current cuda device: {torch.cuda.current_device()}')
    print(f'Count of using GPUs: {torch.cuda.device_count()}')

# argparse dosen't support boolean type
save_model = True if args.save_model == 'True' else False

data = Preprocessing(shuffle=args.shuffle)
num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()

# loading train/test dataframe
train_df, test_df = data.preprocessing()

train_dataset = Tourism(train_df)
test_dataset = Tourism(test_df)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch,
                              shuffle=False,
                              drop_last=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch,
                             shuffle=False,
                             drop_last=True)

model = MatrixFactorization(num_dayofweek = num_dayofweek,
                            num_time = num_time,
                             num_sex = num_sex,
                             num_age = num_age,
                             num_month = num_month,
                             num_day = num_day,
                             num_destination = num_destination,
                             num_dim=8,
                             num_factor=32,)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = RMSELoss()

if __name__ == '__main__' :
    pass