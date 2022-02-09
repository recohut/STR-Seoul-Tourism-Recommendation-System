This repository is for STRMF(Seoul Tourism Recommendation using Matrix Factorization).

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
```bash
python demo.py
device: cpu
몇명이서 관광할 계획이신가요? ex) 3명
2명
몇월 몇일 무슨 요일에 놀러갈 계획이신가요? ex) 1월 3일 수요일
1월 19일 금요일
시간대는 언제가 좋으신가요? ex) 13시
18시
1번째 분의 어떤 연령대 인가요?. ex) 20대
20대
1번째 분의 성별은 무엇이신가요?. ex) 남성/여성
남성
2번째 분의 어떤 연령대 인가요?. ex) 20대
30대
2번째 분의 성별은 무엇이신가요?. ex) 남성/여성
여성

변환된 user info는 다음과 같습니다.

[1, 19, 4, 5, 2029, 1]
[1, 19, 4, 5, 3039, 0]
혼잡도를 고려한 관광지 추천 리스트를 원하시나요?
네
총 몇개의 관광지가 포함된 추천 리스트를 원하시요?
5 개

-------------------Load Destination_info-------------------

Complete Reading Datasets
Load Destination_info complete

-------------------Load Model-------------------

Load Model complete

-------------------1번째 사람을 위한 Top 5등 추천지 입니다.-------------------

1등	 visitor=7.301724910736084	 홍대
2등	 visitor=6.325865745544434	 대학로
3등	 visitor=4.101217269897461	 명동
4등	 visitor=4.083641052246094	 한국종합무역센터(코엑스)
5등	 visitor=3.017594814300537	 인사동

-------------------2번째 사람을 위한 Top 5등 추천지 입니다.-------------------

1등	 visitor=3.723374128341675	 홍대
2등	 visitor=3.557551145553589	 대학로
3등	 visitor=3.0049173831939697	 한국종합무역센터(코엑스)
4등	 visitor=2.9008219242095947	 명동
5등	 visitor=1.8091551065444946	 인사동

------------------- 전체 랭킹리스트에 포함된 관광지 종류 : 5 -------------------

-------------------혼잡도를 고려하지 않은 전체 Top 5등 추천지 입니다.-------------------

1등:누적 visitor=11.02510  누적 congestion=-2.89158   홍대                  
2등:누적 visitor=9.88342   누적 congestion=22.53034   대학로                 
3등:누적 visitor=7.08856   누적 congestion=-11.22470  한국종합무역센터(코엑스)       
4등:누적 visitor=7.00204   누적 congestion=-5.29173   명동                  
5등:누적 visitor=4.82675   누적 congestion=-8.22231   인사동                 

-------------------혼잡도를 고려한 랭킹을 다시 하겠습니다.-------------------
-------------------혼잡도를 고려한 전체 Top 5등 추천지 입니다.-------------------

1등:ndcg variation:14.21506  대학로                 
2등:ndcg variation:-2.27902  명동                  
3등:ndcg variation:-2.89158  홍대                  
4등:ndcg variation:-3.18082  인사동                 
5등:ndcg variation:-5.61235  한국종합무역센터(코엑스)                    

```
