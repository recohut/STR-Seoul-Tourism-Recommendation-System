This repository is for STRMF(Seoul Tourism Recommendation using Matrix Factorization).
STRMF recommends Seoul Tourist destination with visitor & congestion.

Dataset is private.

This project is still under development.

### Quick Start
```bash
cd saved_model
cat visitor_20.z01 visitor_20.z02 visitor_20.z03 visitor_20.z04 visitor_20.zip > Outzip.zip
unzip Outzip.zip
cat congest_20.z01 congest_20.z02 congest_20.z03 congest_20.z04 congest_20.zip > Outzip1.zip
unzip Outzip1.zip
python demo.py
```
### Result

```bash
추천 model을 선택해 주세요. (MF, NGCF, NCF)
MF
device: cpu
몇명이서 관광할 계획이신가요? ex) 3명
2명
몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일
6월 7일 일요일
시간대는 언제가 좋으신가요? ex) 13시
18시
1번째 분의 어떤 연령대 인가요?. ex) 20대
20대
1번째 분의 성별은 무엇이신가요?. ex) 남성/여성
남성
2번째 분의 어떤 연령대 인가요?. ex) 20대
20대
2번째 분의 성별은 무엇이신가요?. ex) 남성/여성
여성

변환된 user info는 다음과 같습니다.

[6, 7, 6, 5, 2029, 1]
[6, 7, 6, 5, 2029, 0]


어떤 장르의 관광지를 원하시나요? (3개 이상 골라주세요) ex) 1,2,3
1.역사관광지 	2.휴양관광지	3.체험관광지	4.문화시설	5.건축/조형물	6.자연관광지	7.쇼핑
2,3,4,7
총 몇개의 관광지가 포함된 추천 리스트를 원하시요?
10개
어디서 출발하시나요? 행정구와 동을 입력해주세요. ex) 종로구 삼청동
종로구 삼청동
여행 계획이 총 몇일 이신 가요? ex) 3일
3일
세개의 요소 [선호도, 혼잡도, 거리]에 대한 가중치는 각각 어떻게 둘까요? ex) 0.5,0.3,0.2 
0.5, 0.3, 0.2

변환된 user info는 다음과 같습니다.
[6, 7, 6, 5, 2029, 1]
[6, 7, 6, 5, 2029, 0]
[6, 8, 0, 5, 2029, 1]
[6, 8, 0, 5, 2029, 0]
[6, 9, 1, 5, 2029, 1]
[6, 9, 1, 5, 2029, 0]
Loading Dataset: 100%|██████████████████████████████████████████████| 10/10 [00:01<00:00,  9.83it/s]

Load Destination_info complete

Loading Model: 100%|████████████████████████████████████████████████| 10/10 [00:01<00:00,  9.81it/s]
Load Model complete


1일 째 추천 관광지
1등: 롯데월드
2등: 겸재정선미술관
3등: 올림픽공원
4등: 전쟁기념관
5등: 서울역사박물관
6등: 천호공원
7등: 명동
8등: 도산공원
9등: 한강시민공원 뚝섬지구(뚝섬한강공원)
10등: 월드컵공원

2일 째 추천 관광지
1등: 신사동 가로수길
2등: 망원시장
3등: 대학로
4등: 한강시민공원 여의도지구(여의도한강공원)
5등: 영등포 신길동 홍어거리
6등: 청와대 앞길
7등: 보라매공원
8등: 인사동
9등: 서울시립 북서울미술관
10등: 살곶이체육공원

3일 째 추천 관광지
1등: 압구정 로데오거리
2등: 노량진수산물도매시장
3등: 서래마을
4등: 양재 시민의숲
5등: 온수공원
6등: 서울어린이대공원
7등: 파리공원
8등: 둘리뮤지엄
9등: 자양동 양꼬치거리 (중국음식문화거리)
10등: 응암동 감자국 거리
추천하는데 총 걸린 시간 : 3.972867965698242                        

```
