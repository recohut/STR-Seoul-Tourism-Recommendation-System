This repository is for STRMF(Seoul Tourism Recommendation using Matrix Factorization).
STRMF recommends Seoul Tourist destination with visitor & congestion.

Dataset is private.

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
!(image)[https://imgur.com/Bp1xMwL]

```bash
python demo.py
device: cpu
몇명이서 관광할 계획이신가요? ex) 3명
3명
몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일
1월 27일 수요일
시간대는 언제가 좋으신가요? ex) 13시
15시
1번째 분의 어떤 연령대 인가요?. ex) 20대
20대
1번째 분의 성별은 무엇이신가요?. ex) 남성/여성
남성
2번째 분의 어떤 연령대 인가요?. ex) 20대
30대
2번째 분의 성별은 무엇이신가요?. ex) 남성/여성
여성
3번째 분의 어떤 연령대 인가요?. ex) 20대
40대
3번째 분의 성별은 무엇이신가요?. ex) 남성/여성
남성

변환된 user info는 다음과 같습니다.

[1, 27, 2, 4, 2029, 1]
[1, 27, 2, 4, 3039, 0]
[1, 27, 2, 4, 4049, 1]
혼잡도를 고려한 관광지 추천 리스트를 원하시나요?
네
총 몇개의 관광지가 포함된 추천 리스트를 원하시요?
7개

-------------------Load Destination_info-------------------

Complete Reading Datasets
Load Destination_info complete

-------------------Load Model-------------------

Load Model complete

-------------------1번째 사람을 위한 Top 7등 추천지 입니다.-------------------

1등      visitor=7.153442859649658       홍대
2등      visitor=6.484706878662109       대학로
3등      visitor=5.924557685852051       한국종합무역센터(코엑스)
4등      visitor=5.453670978546143       명동
5등      visitor=4.402748107910156       인사동
6등      visitor=3.1326382160186768      압구정 로데오거리
7등      visitor=3.064981460571289       신사동 가로수길

-------------------2번째 사람을 위한 Top 7등 추천지 입니다.-------------------

1등      visitor=4.845834732055664       한국종합무역센터(코엑스)
2등      visitor=4.253275394439697       명동
3등      visitor=3.7163922786712646      대학로
4등      visitor=3.5750911235809326      홍대
5등      visitor=3.194307565689087       인사동
6등      visitor=2.9512557983398438      남대문시장
7등      visitor=2.154278516769409       압구정 로데오거리

-------------------3번째 사람을 위한 Top 7등 추천지 입니다.-------------------

1등      visitor=4.201891899108887       한국종합무역센터(코엑스)
2등      visitor=3.2735886573791504      인사동
3등      visitor=3.2653281688690186      명동
4등      visitor=3.148672103881836       남대문시장
5등      visitor=2.286957263946533       대학로
6등      visitor=2.010316848754883       서울역사박물관
7등      visitor=1.965502381324768       경복궁

------------------- 전체 랭킹리스트에 포함된 관광지 종류:10-------------------

-------------------혼잡도를 고려하지 않은 전체 Top 7등 추천지 입니다.-------------------

1등:누적 visitor=14.97228  누적 congestion=2.78067    한국종합무역센터(코엑스)       
2등:누적 visitor=12.97227  누적 congestion=7.77371    명동                  
3등:누적 visitor=12.48806  누적 congestion=1.97151    대학로                 
4등:누적 visitor=10.87064  누적 congestion=0.98326    인사동                 
5등:누적 visitor=10.72853  누적 congestion=4.86175    홍대                  
6등:누적 visitor=6.09993   누적 congestion=3.81239    남대문시장               
7등:누적 visitor=5.28692   누적 congestion=1.85461    압구정 로데오거리           

-------------------혼잡도를 고려한 랭킹을 다시 하겠습니다.-------------------
-------------------혼잡도를 고려한 전체 Top 7등 추천지 입니다.-------------------

1등:ndcg varation:0.43801   인사동                 
2등:ndcg varation:0.35963   한국종합무역센터(코엑스)       
3등:ndcg varation:0.25361   대학로                 
4등:ndcg varation:0.17973   압구정 로데오거리           
5등:ndcg varation:0.09343   남대문시장               
6등:ndcg varation:0.08116   명동                  
7등:ndcg varation:0.07957   홍대                             

```
