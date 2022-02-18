'''
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer


'''
    date : 2018-01-01 ~ 2020-12-31
    destination : 관광지 코드
    time : 방문한 시간대
    sex : 성
    age : 나이
    visitor : 방문객수
    year : 년도
    month : 월
    day : 일
    dayofweek : 요일
    total_num : 총 수용인원수
    area : 관광지 면적
    date365 : 0~365
    congestion_1 : 방문객수 / 총수용인원수
    congestion_2 : 방문객수 / 관광지면적
'''

class Preprocessing():
    def __init__(self, shuffle=True,use_demo=False):
        self.shuffle = shuffle
        self.use_demo = use_demo
        if self.use_demo:
            self.destination_id_name_df = self._read_Destination_info()
        else:
            self.merged_df, self.destination_id_name_df = self._read_dataset(),self._read_Destination_info()
        print("Complete Reading Datasets")

    def _read_dataset(self):
        # merged_df = pd.read_csv("../Preprocessing/Datasets_v3.1/Datasets_v3.1.txt", sep='|')
        merged_df = pd.read_csv("dataset/Datasets_v3.1.txt", sep='|')
        return merged_df

    def _read_Destination_info(self):
        # destination_id_name_df = pd.read_csv("../Preprocessing/Datasets_v3.1/destination_id_name.csv")
        destination_id_name_df = pd.read_csv('dataset/destination_id_name.csv')
        return destination_id_name_df

    def get_num(self):
        merged_df = self.merged_df.copy()
        num_destination = merged_df['destination'].max()+1
        num_time = merged_df['time'].max()+1
        num_sex = merged_df['sex'].max()+1
        num_age = merged_df['age'].max()+1
        num_dayofweek = merged_df['dayofweek'].max()+1
        num_month = merged_df['month'].max()+1
        num_day = merged_df['day'].max()+1
        return num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day

    def preprocessing(self):
        scaler = StandardScaler()
        total_df = self.merged_df.copy()

        # congetion^-1
        # total_df[['congestion_1','congestion_2']] = 1/total_df[['congestion_1','congestion_2']]
        # df2018[['congestion_1','congestion_2']] = 1/df2018[['congestion_1','congestion_2']]
        # df2019[['congestion_1','congestion_2']] = 1/df2019[['congestion_1','congestion_2']]
        # df2020[['congestion_1','congestion_2']] = 1/df2020[['congestion_1','congestion_2']]

        # congestion normalize & train test split
        if self.shuffle == 0:
            total_df[['congestion_1','congestion_2','visitor']] = scaler.fit_transform(pd.DataFrame(total_df[['congestion_1','congestion_2','visitor']]))
            df2018 = total_df[total_df['year']==2018]
            df2019 = total_df[total_df['year']==2019]
            df2020 = total_df[total_df['year']==2020]
            train_df = df2018
            test_df = df2019
            print("Complete Normalize Datasets")
        else:
            total_df[['congestion_1','congestion_2','visitor']] = scaler.fit_transform(pd.DataFrame(total_df[['congestion_1','congestion_2','visitor']]))
        print(f'len(Train dataframe): {len(train_df)}, \t len(Test dataframe): {len(test_df)}')
        print("Complete Train Test Split")
        return train_df, test_df

    def destination_list(self,genre_list):
        des_list = list(self.destination_id_name_df['destination'].unique())
        return self.destination_id_name_df, des_list


class Tourism(Dataset):
    def __init__(self, df, target_name):
        super(Tourism, self).__init__()
        self.df = df
        self.target_name = target_name
        self.destination, self.time, self.sex, self.age, self.dayofweek, self.month, self.day, self.target = self.change_tensor()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.destination[idx], self.time[idx], self.sex[idx], self.age[idx], \
               self.dayofweek[idx], self.month[idx], self.day[idx], self.target[idx]

    def change_tensor(self):
        target_name = self.target_name
        destination = torch.tensor(list(self.df['destination']))
        time = torch.tensor(list(self.df['time']))
        sex = torch.tensor(list(self.df['sex']))
        age = torch.tensor(list(self.df['age']))
        dayofweek = torch.tensor(list(self.df['dayofweek']))
        month = torch.tensor(list(self.df['month']))
        day = torch.tensor(list(self.df['day']))
        target = torch.tensor(list(self.df[target_name]))
        return destination, time, sex, age, dayofweek, month, day, target


class Input_Dataset(Dataset):
    def __init__(self, destination_list, RecSys_input):
        super(Input_Dataset, self).__init__()
        self.destination_list = destination_list
        self.num_dest = len(destination_list)
        self.RecSys_input = RecSys_input
        self.month, self.day, self.dayofweek, self.time, self.sex, self.age, self.destination = self.change_tensor()

    def __len__(self):
        return len(self.month)

    def __getitem__(self, idx):
        return self.destination[idx], self.time[idx], self.sex[idx], self.age[idx], self.dayofweek[idx], self.month[idx], self.day[idx]

    def change_tensor(self):
        month = torch.tensor([self.RecSys_input[0]] * self.num_dest)
        day = torch.tensor([self.RecSys_input[1]] * self.num_dest)
        dayofweek = torch.tensor([self.RecSys_input[2]] * self.num_dest)
        time = torch.tensor([self.RecSys_input[3]] * self.num_dest)
        age = torch.tensor([self.RecSys_input[4]] * self.num_dest)
        sex = torch.tensor([self.RecSys_input[5]] * self.num_dest)
        destination = torch.tensor(self.destination_list)
        return month, day, dayofweek, time, sex, age, destination
