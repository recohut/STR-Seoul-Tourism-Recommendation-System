from tqdm import tqdm
import time
import os
import pandas as pd
# ROOT_DIR = 'dataset'
# DEST_INFO_PATH = os.path.join(ROOT_DIR, 'destination_id_name_genre_coordinate.csv')
# PREICTED_CONGEST_PATH = os.path.join(ROOT_DIR, 'congestion_1_2.csv')
# CITY_INFO_PATH = os.path.join(ROOT_DIR, 'seoul_gu_dong_coordinate.csv')
#
# li = [DEST_INFO_PATH,PREICTED_CONGEST_PATH,CITY_INFO_PATH]
# for i in li:
#     df = pd.read_csv(i)
#     df.to_pickle(i.rstrip('csv')+'pkl')
#
#
# destination_id_name_df, destination_list = filter_destination(DEST_INFO_PATH, genre_list)
# batch_candidate = len(destination_list)
# congestion_df = pd.read_csv(PREICTED_CONGEST_PATH)
# city_df = pd.read_csv(CITY_INFO_PATH)
# start_df = city_df[(city_df['gu'] == start_info[0]) & (city_df['dong'] == start_info[1])]
# user_pos = (start_df['y'], start_df['x'])
# st= time.time()
# df1= pd.read_csv(PREICTED_CONGEST_PATH)
# print(time.time()-st)
# df1.to_pickle('test.pkl')
chunks = []
text= 'hi'
t = tqdm(total=10, ncols=100,desc=text)
for i in range(5):
    time.sleep(0.1)
    t.update(2)
t.close()

df1= pd.read_pickle('test.pkl')

print(time.time()-st2)