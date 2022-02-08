import os
import sys
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

def input_filterchar(userinfo:str):
    str=""
    for token in userinfo:
        if ord(token)<48 or ord(token)>57:
            break
        str+=token
    return int(str)

def char2num(chr):
    day_dict ={'월':0,'화':1,'수':2,'목':3,'금':4,'토':5,'일':6}
    return day_dict[chr]

def str2datetime(date_info:list):
    month, day, dayofweek = None, None, None
    for i,token in enumerate(date_info):
        str = ""
        if i==0 :
            month = int(input_filterchar(token))
        if i==1:
            day = int(input_filterchar(token))
        if i==2:
            for c in token:
                if c!='요':
                    str+=c
                else:
                    break
            dayofweek = char2num(str)
    return month, day, dayofweek

def time2range(time):
    if time<6:
        return 1
    if time<11:
        return 2
    if time<14:
        return 3
    if time<18:
        return 4
    if time<21:
        return 5
    if time<24:
        return 6

def age2range(age):
    return (age//10)*10+5

def sex2int(sex):
    return 1 if sex=='남성'else 0

if __name__ == '__main__' :
    # print("몇명이서 관광할 계획이신가요? ex) 3명")
    # num_people = input_filterchar(input())
    # print("몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일")
    # date = list(input().split())
    # print("시간대는 언제가 좋으신가요? ex) 13시")
    # timezone = time2range(input_filterchar(input()))
    # month, day, dayofweek = str2datetime(date)
    # total_user_info =[]
    # for i in range(1, num_people + 1):
    #     li=[month,day,dayofweek,timezone]
    #     print(f"{i}번째 분의 어떤 연령대 인가요?. ex) 20대")
    #     li.append(input_filterchar(input()))
    #     print(f"{i}번째 분의 성별은 무엇이신가요?. ex) 남성/여성")
    #     li.append(sex2int(input()))
    #     total_user_info.append(li)
    #
    # print("변환된 user info는 다음과 같습니다.")
    # for i in total_user_info:
    #     print(i)
    # RecSys_input = total_user_info
    RecSys_total_input =[]
    with open('sample_input.txt',mode='r') as f:
        for line in f:
            RecSys_total_input.append([int(x) for x in line.split(',')])
    print(RecSys_total_input)

    print("-------------------Load Destination_info-------------------")
    data = Preprocessing(shuffle=False)
    destination_id_name_df, destination_list = data.destination_list()
    batch_candidate = 100
    print("Load Destination_info complete")

    print("-------------------Load Model-------------------")
    model_root_dir ='saved_model'
    model_dir = os.path.join(model_root_dir,'MF_6.pth')
    if not os.path.exists(model_dir):
        print("Model doesn't exist.")
        sys.exit()
    model = torch.load(model_dir)
    print("Load Model complete")

    for i,user_input in enumerate(RecSys_total_input):
        RecSys_dataset = Input_Dataset(destination_list=destination_list, RecSys_input=user_input)
        RecSys_dataloader = DataLoader(dataset=RecSys_dataset, batch_size=batch_candidate, shuffle=False)
        for month, day, dayofweek, time, sex, age, destination in RecSys_dataloader:
            # itemId
            destination = destination.to(device)
            # user information(userId)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device), age.to(
                device), month.to(device), day.to(device)
            prediction = model(dayofweek, time, sex, age, month, day, destination)

        prediction = prediction.view(-1)
        prediction = prediction.tolist()
        destination_id_name_df['pred_congestion'] = prediction
        destination_id_name_df = destination_id_name_df.sort_values(by='pred_congestion', ascending=False)

        for i in range(len(destination_id_name_df)):
            print("{}\t{}".format(destination_id_name_df.iloc[i, 2], destination_id_name_df.iloc[i, 1]))
