import os
import random
import torch
import warnings
from torch.utils.data import DataLoader
import torch.optim as optim
from parser import args
from utils import Tourism, Preprocessing, Input_Dataset
from datetime import datetime
from model.MF import MatrixFactorization
from criterion import RMSELoss
from train import train



# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device: {device}')

# print GPU information
if torch.cuda.is_available():
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

warnings.filterwarnings('ignore')

# argparse dosen't support boolean type
save_model = True if args.save_model == 'True' else False
use_shuffle = True if args.shuffle =='True' else False

# select target
target_name = 'visitor' if args.target == 'v' else 'congestion_1'
print(f'selected target is {target_name}')

# shuffle
data = Preprocessing(shuffle=use_shuffle)
num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()

# loading train/test dataframe
train_df, test_df = data.preprocessing()
train_dataset = Tourism(train_df,target_name)
test_dataset = Tourism(test_df,target_name)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=False,
                              drop_last=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
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
                             num_factor=48,)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = RMSELoss()

if __name__ == '__main__' :
    print('-------------------Train Start-------------------')
    start_time = datetime.now()

    train(model=model,
          optimizer=optimizer,
          epochs=args.epochs,
          dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          criterion=criterion,
          device=device,
          print_cost=True)

    end_time = datetime.now()
    print('-------------------Train Finished-------------------')
    print(f'Training time : {end_time -start_time}')


    FOLDER_PATH ='saved_model'
    if not os.path.exists(FOLDER_PATH):
        os.mkdir(FOLDER_PATH)

    if save_model:
        MODEL_PATH = os.path.join(FOLDER_PATH, f'MF_{args.epochs}_{args.batch_size}_{target_name}.pth')
        torch.save(model.state_dict(), MODEL_PATH)

