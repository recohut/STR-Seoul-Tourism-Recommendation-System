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
if device == 'cuda':
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())

warnings.filterwarnings('ignore')

# argparse dosen't support boolean type
save_model = True if args.save_model == 'True' else False
use_shuffle = True if args.shuffle =='True' else False

# shuffle
data = Preprocessing(shuffle=use_shuffle)
num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()

# loading train/test dataframe
train_df, test_df = data.preprocessing()
train_dataset = Tourism(train_df)
test_dataset = Tourism(test_df)

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
                             num_dim=4,
                             num_factor=32,)

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


    model_root_dir ='saved_model'
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    model_id = random.randrange(1,101)
    if save_model:
        model_dir = os.path.join(model_root_dir, f'MF_{model_id}+.pth')
        while (1):
            if os.path.exists(model_dir):
                model_id = random.randrange(1, 101)
                model_dir = os.path.join(model_root_dir, f'MF_{model_id}.pth')
            else:
                break
        torch.save(model, model_dir)
