'''
code from IkGyu-Lee
link : https://github.com/IkGyu-Lee
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
    def __init__(self, shuffle=True):
        self.shuffle = shuffle
        self.merged_df, self.destination_id_name_df = self.read_dataset()

    def read_dataset(self):
        # Dataframe 불러오기
        # merged_df = pd.read_csv("../Preprocessing/Datasets_v3.1/Datasets_v3.1.txt", sep='|')
        # destination_id_name_df = pd.read_csv("../Preprocessing/Datasets_v3.1/destination_id_name.csv")
        merged_df = pd.read_csv("dataset/Datasets_v3.1.txt", sep='|')
        destination_id_name_df = pd.read_csv('dataset/destination_id_name.csv')
        print("Complete Reading Datasets")
        return merged_df, destination_id_name_df

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
        df2018 = total_df[total_df['year']==2018]
        df2019 = total_df[total_df['year']==2019]
        df2020 = total_df[total_df['year']==2020]

        # congetion^-1
        total_df[['congestion_1','congestion_2']] = 1/total_df[['congestion_1','congestion_2']]
        df2018[['congestion_1','congestion_2']] = 1/df2018[['congestion_1','congestion_2']]
        df2019[['congestion_1','congestion_2']] = 1/df2019[['congestion_1','congestion_2']]
        df2020[['congestion_1','congestion_2']] = 1/df2020[['congestion_1','congestion_2']]

        # congestion normalize & train test split
        if self.shuffle == False:
            train_df = df2018
            test_df = df2019
            train_df[['congestion_1','congestion_2']] = scaler.fit_transform(pd.DataFrame(train_df[['congestion_1','congestion_2']]))
            test_df[['congestion_1','congestion_2']] = scaler.fit_transform(pd.DataFrame(test_df[['congestion_1','congestion_2']]))
            print("Complete Normalize Datasets")
        else:
            total_df[['congestion_1','congestion_2']] = scaler.fit_transform(pd.DataFrame(total_df[['congestion_1','congestion_2']]))
            print("Complete Normalize Datasets")
            train_df, test_df, y_train, y_test = train_test_split(total_df, total_df['destination'], test_size=0.3, stratify=total_df['destination'], random_state=42)

        print("Complete Train Test Split")
        return train_df, test_df

    def destination_list(self):
        des_list = list(self.destination_id_name_df['destination'].unique())
        return self.destination_id_name_df, des_list


class Tourism(Dataset):
    def __init__(self, df):
        super(Tourism, self).__init__()
        self.df = df
        self.destination, self.time, self.sex, self.age, self.dayofweek, self.month, self.day, self.congestion_1 = self.change_tensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.destination[idx], self.time[idx], self.sex[idx], self.age[idx], self.dayofweek[idx], self.month[idx], self.day[idx], self.congestion_1[idx]

    def change_tensor(self):
        destination = torch.tensor(list(self.df['destination']))
        time = torch.tensor(list(self.df['time']))
        sex = torch.tensor(list(self.df['sex']))
        age = torch.tensor(list(self.df['age']))
        dayofweek = torch.tensor(list(self.df['dayofweek']))
        month = torch.tensor(list(self.df['month']))
        day = torch.tensor(list(self.df['day']))
        congestion_1 = torch.tensor(list(self.df['congestion_1']))
        return destination, time, sex, age, dayofweek, month, day, congestion_1


class Input_Dataset(Dataset):
    def __init__(self, destination_list, RecSys_input):
        super(Input_Dataset, self).__init__()
        self.destination_list = destination_list
        self.RecSys_input = RecSys_input
        self.month, self.day, self.dayofweek, self.time, self.sex, self.age, self.destination = self.change_tensor()

    def __len__(self):
        return len(self.month)

    def __getitem__(self, idx):
        return self.month[idx], self.day[idx], self.dayofweek[idx], self.time[idx], self.sex[idx], self.age[idx], self.destination[idx]

    def change_tensor(self):
        month = torch.tensor([self.RecSys_input[0]] * 100)
        day = torch.tensor([self.RecSys_input[1]] * 100)
        dayofweek = torch.tensor([self.RecSys_input[2]] * 100)
        time = torch.tensor([self.RecSys_input[3]] * 100)
        sex = torch.tensor([self.RecSys_input[4]] * 100)
        age = torch.tensor([self.RecSys_input[5]] * 100)
        destination = torch.tensor(self.destination_list)

        return month, day, dayofweek, time, sex, age, destination
