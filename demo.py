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
    return ((age//10)*10)*100+ ((age//10)*10+9)

def sex2int(sex):
    return 1 if sex[0]=='남'else 0

if __name__ == '__main__' :
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

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
    #     li.append(age2range(input_filterchar(input())))
    #     print(f"{i}번째 분의 성별은 무엇이신가요?. ex) 남성/여성")
    #     li.append(sex2int(input()))
    #     total_user_info.append(li)
    # RecSys_total_input = total_user_info
    print("변환된 user info는 다음과 같습니다.\n")
    RecSys_total_input =[]
    with open('sample_input.txt',mode='r') as f:
        for line in f:
            RecSys_total_input.append([int(x) for x in line.split(',')])
    for i in RecSys_total_input:
        print(i)

    print("-------------------Load Destination_info-------------------\n")
    data = Preprocessing(shuffle=False)
    num_destination, num_time, num_sex, num_age, num_dayofweek, num_month, num_day = data.get_num()
    destination_id_name_df, destination_list = data.destination_list()
    batch_candidate = 100
    print("Load Destination_info complete\n")
    print("-------------------Load Model-------------------\n")
    FOLDER_PATH ='saved_model'
    MODEL_PATH = os.path.join(FOLDER_PATH,'MF_20_256_diag_no_MLP.pth')
    if not os.path.exists(MODEL_PATH):
        print("Model doesn't exist.\n")
        sys.exit()

    model = MatrixFactorization(num_dayofweek=num_dayofweek,
                                num_time=num_time,
                                num_sex=num_sex,
                                num_age=num_age,
                                num_month=num_month,
                                num_day=num_day,
                                num_destination=num_destination,
                                num_dim=8,
                                num_factor=48, )
    model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
    print("Load Model complete\n")

    topk = 10
    total_ranking = {}

    for i,user_input in enumerate(RecSys_total_input):
        user_df = destination_id_name_df
        RecSys_dataset = Input_Dataset(destination_list=destination_list, RecSys_input=user_input)
        RecSys_dataloader = DataLoader(dataset=RecSys_dataset,
                                       batch_size=batch_candidate,
                                       shuffle=False)
        for destination, time, sex, age, dayofweek, month, day  in RecSys_dataloader:
            destination = destination.to(device)
            dayofweek, time, sex, age, month, day = dayofweek.to(device), time.to(device), sex.to(device), age.to(
                device), month.to(device), day.to(device)
            pred = model(dayofweek, time, sex, age, month, day, destination)
        pred = pred.view(-1)
        pred = pred.tolist()
        user_df['pred_congestion'] = pred
        user_df = user_df.sort_values(by='pred_congestion', ascending=False)

        print(f'\n-------------------{i+1}번째 사람을 위한 Top {topk}등 추천지 입니다.-------------------\n')

        for k in range(topk):
            destionation_name = user_df.iloc[k, 1]
            pred_target = user_df.iloc[k, 2]
            print(f'{k+1}등:\t{pred_target}\t{destionation_name}')

            if(rank_weight := total_ranking.get(destionation_name)) is None:
                total_ranking[destionation_name]=0
            total_ranking[destionation_name]+=topk-k

    print(f'-------------------전체 Top {topk}등 추천지 입니다.-------------------\n')
    sorted_total_ranking = sorted(total_ranking.items(), key=lambda item:item[1], reverse=True)
    print("전체 랭킹리스트 개수: ",len(sorted_total_ranking))
    for k in range(topk):
        print(f'{k+1}등 :\t{sorted_total_ranking[k]}')

