'''
hj04143@gmail.com
https://github.com/changhyeonnam/STRMF
'''
# -*- coding: utf-8 -*-
import os
import sys
import random
import torch
import warnings
from torch.utils.data import DataLoader
from utils import Tourism, Preprocessing, Input_Dataset
from model.MF import MatrixFactorization
import numpy as np
import pandas as pd
import time as ti
from haversine import haversine
import math
import tqdm
from parser import args
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

def destint2str(li):
    dest_dict = {'1':'역사관광지','2':'휴양관광지','3':'체험관광지','4':'문화시설','5':'건축/조형물','6':'자연관광지','7':'쇼핑'}
    dest_li=[]
    for val in li:
        dest_li.append(dest_dict[val])
    return dest_li

def load_congestion(destination_id_name_df,df,dayofweek, time, month, day):
    dayofweek, time, day, month = dayofweek.item(), time.item(), day.item(), month.item()
    new_df = df[((df['month'] ==month) & (df['day']==day)) & ((df['dayofweek']==dayofweek) & (df['time'] ==time))]

    new_df = new_df[new_df.destination.isin(destination_id_name_df.destination)]

    return new_df['congestion_1']

def filter_destination(DEST_PATH,genre_list):
    df = pd.read_pickle(DEST_PATH)
    newdf = pd.DataFrame(columns=['destination', 'destination_name', 'large_category', 'middle_category',
                                  'small_category', 'large_category_name', 'middle_category_name',
                                  'small_category_name', 'x', 'y'])
    for i in genre_list:
        ndf = df[(df['middle_category_name'] == i)]
        newdf = pd.concat([newdf, ndf],ignore_index=True)
    des_list = newdf['destination'].to_list()
    return newdf,des_list

def data_sampling(data, period):
    month_dict = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:30}
    new_data = data.copy()
    period-=1
    for i in range(1,period+1):
        for each in data:
            tmp = each.copy()
            tmp[1] = tmp[1]+i
            tmp[2] = (tmp[2]+i )%7
            if(month_dict[tmp[0]]<tmp[1]):
                tmp[1]=1
                tmp[0] = tmp[0]+1
                if tmp[0]==13:
                    tmp[0]=1
            new_data.append(tmp)
    return new_data



# just for visualization for laoding
def progress_bar(text):
    ti.sleep(0.01)
    t = tqdm.tqdm(total=10, ncols=100, desc=text)
    for i in range(5):
        ti.sleep(0.2)
        t.update(2)
    t.close()

def MF_demo():
    # check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    # print GPU information
    if torch.cuda.is_available():
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())

    print("몇명이서 관광할 계획이신가요? ex) 3명")
    num_people = input_filterchar(input())
    print("몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일")
    date = list(input().split())
    print("시간대는 언제가 좋으신가요? ex) 13시")
    timezone = time2range(input_filterchar(input()))
    month, day, dayofweek = str2datetime(date)
    total_user_info =[]
    for i in range(1, num_people + 1):
        li=[month,day,dayofweek,timezone]
        print(f"{i}번째 분의 어떤 연령대 인가요?. ex) 20대")
        li.append(age2range(input_filterchar(input())))
        print(f"{i}번째 분의 성별은 무엇이신가요?. ex) 남성/여성")
        li.append(sex2int(input()))
        total_user_info.append(li)
    RecSys_total_input = total_user_info


    # print converted user info
    print("\n변환된 user info는 다음과 같습니다.\n")
    for i in RecSys_total_input:
        print(i)
    print('\n')
    # select destination genre
    print("어떤 장르의 관광지를 원하시나요? (3개 이상 골라주세요) ex) 1,2,3"
          "\n1.역사관광지 \t2.휴양관광지\t3.체험관광지\t4.문화시설\t5.건축/조형물\t6.자연관광지\t7.쇼핑")
    genre_list = destint2str(input().split(','))


    # check for congestion
    # print("혼잡도를 고려한 관광지 추천 리스트를 원하시나요?")
    # check_congestion = True if input()=='네' else False

    # input for topk
    print("총 몇개의 관광지가 포함된 추천 리스트를 원하시요?")
    topk = input_filterchar(input())

    # input for staring point
    print("어디서 출발하시나요? 행정구와 동을 입력해주세요. ex) 종로구 삼청동")
    start_info = input().split(' ')
    print("여행 계획이 총 몇일 이신 가요? ex) 3일")
    duration = input_filterchar(input())
    print('세개의 요소 [선호도, 혼잡도, 거리]에 대한 가중치는 각각 어떻게 둘까요? ex) 0.5,0.3,0.2 ')
    weight = list(map(float,input().split(',')))
    # weight=[0.6,0.2,0.2]
    # start_info=['종로구', '삼청동']
    # check_congestion = True
    # topk = 10
    # duration=3
    # # for sample testcase
    # RecSys_total_input =[]
    # with open('sample_input.txt',mode='r') as f:
    #     for line in f:
    #         RecSys_total_input.append([int(x) for x in line.split(',')])
    # num_people=len(RecSys_total_input)
    #
    # genre_list =destint2str('2,3,4,7'.split(','))

    RecSys_total_input = data_sampling(RecSys_total_input,duration)
    print("\n변환된 user info는 다음과 같습니다.")
    for i in RecSys_total_input:
        print(i)
    total_start = ti.time()

    progress_bar('Loading Dataset')
    ROOT_DIR = 'dataset'
    DEST_INFO_PATH = os.path.join(ROOT_DIR,'destination_id_name_genre_coordinate.pkl')
    PREICTED_CONGEST_PATH = os.path.join(ROOT_DIR,'congestion_1_2.pkl')
    CITY_INFO_PATH = os.path.join(ROOT_DIR,'seoul_gu_dong_coordinate.pkl')

    destination_id_name_df, destination_list = filter_destination(DEST_INFO_PATH,genre_list)
    batch_candidate = len(destination_list)
    congestion_df = pd.read_pickle(PREICTED_CONGEST_PATH)
    city_df = pd.read_pickle(CITY_INFO_PATH)
    start_df = city_df[(city_df['gu'] == start_info[0]) & (city_df['dong'] == start_info[1])]
    user_pos = (start_df['y'], start_df['x'])

    # print(f"Load Destination_info complete\t{ti.time()-st1}\n")
    print(f"\nLoad Destination_info complete\n")
    progress_bar('Loading Model')
    # st2 = ti.time()

    FOLDER_PATH ='saved_model'
    MODEL_PATH_VISITOR = os.path.join(FOLDER_PATH,'MF_20_256_visitor.pth')
    if not os.path.exists(MODEL_PATH_VISITOR) :
        print("Model doesn't exist.\n")
        sys.exit()

    model_visitor = MatrixFactorization(num_dayofweek=7,
                                num_time=7,
                                num_sex=2,
                                num_age=7001,
                                num_month=13,
                                num_day=32,
                                num_destination=2505928,
                                num_dim=8,
                                num_factor=48, )


    model_visitor.load_state_dict(torch.load(MODEL_PATH_VISITOR,map_location=device))
    # print(f"Load Model complete\t{ti.time()-st2}\n")
    print(f"Load Model complete\n")

    All_day_ranking=[0]*(duration)
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
            pred_visitor = model_visitor(dayofweek, time, sex, age, month, day, destination)
            saved_congestion = load_congestion(destination_id_name_df,df=congestion_df,dayofweek=dayofweek[0], time=time[0],month= month[0],day= day[0],)
            pred_visitor = pred_visitor.tolist()
            saved_congestion = saved_congestion.tolist()
            user_df['visitor'] = pred_visitor
            user_df['congestion'] = saved_congestion
            user_df = user_df.sort_values(by='visitor', ascending=False)

            # print(f'\n-------------------{i+1}번째 사람을 위한 Top {topk}등 추천지 입니다.-------------------\n')
            for k in range(batch_candidate):
                dest_pos = (user_df.iloc[k,8],user_df.iloc[k,9])
                destionation_name = user_df.iloc[k,1]
                small_genre = user_df.iloc[k,7]
                middle_genre= user_df.iloc[k,6]
                pred_visitor = user_df.iloc[k,10]
                pred_congestion = user_df.iloc[k,11]
                distance = haversine(user_pos,dest_pos)
                # print(f'{k+1}등\tvisitor={pred_visitor}\t {destionation_name}')
                # print(f'{k+1}등\tmiddle_genre:{middle_genre}\tsmall_genre:{small_genre}\tdistance:{distance}')
                rank_weight = total_ranking.get(destionation_name)
                if rank_weight is None:
                    total_ranking[destionation_name]=[0,0,distance,middle_genre,small_genre]
                total_ranking[destionation_name][0]+=pred_visitor
                total_ranking[destionation_name][1]+=pred_congestion
        if i%num_people == num_people-1:
            All_day_ranking[i//num_people]=total_ranking
            total_ranking={}

    result_ranking=[0]*duration
    total_dest={}
    dest_list = []
    for day,each_rank in enumerate(All_day_ranking):
        sorted_visitor_ranking = sorted(each_rank.items(),key=lambda item:item[1][0],reverse=True)
        sorted_congestion_ranking = sorted(each_rank.items(),key=lambda item:item[1][1],reverse=True)
        sorted_distance_ranking = sorted(each_rank.items(),key=lambda item:item[1][2],reverse=True)
        tmp_rank={}
        for i,dest in enumerate(sorted_visitor_ranking):
            tmp_rank[dest[0]] = weight[0]*(batch_candidate-i)
        for i,dest in enumerate(sorted_congestion_ranking):
            tmp_rank[dest[0]] += weight[1]*(batch_candidate-i)
        for i,dest in enumerate(sorted_distance_ranking):
            tmp_rank[dest[0]] += weight[2]*(batch_candidate-i)
            tmp_rank[dest[0]] = round(tmp_rank[dest[0]],2)
            # for median value
            val = total_dest.get(dest[0])
            if val is None:
                total_dest[dest[0]]=[]
            total_dest[dest[0]].append(tmp_rank[dest[0]])

        sorted_total_ranking = sorted(tmp_rank.items(), key=lambda item:item[1],reverse=True)
        result_ranking[day] = sorted_total_ranking

    # caluate median ranknig
    median_dest={}
    for k,v in total_dest.items():
        v.sort(reverse=True)
        median_dest[k]=v[len(v)//2]

    for i,day_ranking in enumerate(result_ranking):
        print(f'\n{i+1}일 째 추천 관광지')
        rank = 1
        for dest in day_ranking:
            if median_dest[dest[0]]>=dest[1]:
                continue
            print(f'{rank}등: {dest[0]}')
            rank+=1
            if rank==11:
                break

    end_time = ti.time()
    print(f'추천하는데 총 걸린 시간 : {end_time-total_start}')