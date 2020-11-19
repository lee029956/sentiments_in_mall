# AI를 이용한 쇼핑몰 댓글 감정분석<br>
<br>

## 목차<br>
개요<br>
프로세스<br>
데이터수집(web-crawling)<br>
Labeling 및 핵심문장 추출<br>
핵심단어 저장(one-hot)<br>
학습 및 결과확인<br>
실행순서<br>
Prerequisites<br>
<br>

## 개요<br>
온라인 쇼핑몰에서 어떤 상품이 잘 팔리는지? 품질이 좋은지? 전반적으로 만족스러운지? 구입 전에 유의해야 할 부분은 없는지? 쉽게 알 수 있는 방법은 없을까?<br>
<br>
또, 회사에서도 신상품 반응이 어떤지? 고객들은 만족하고 있는지? 보완해야 할 부분은 없는지? 파악할 수 있는 방법이 없을까?<br>
<br>
AI로 상품평(댓글)을 분석하여 소비자의 감성이 어떠한지 파악해 보도록 하자<br>
<br>

## 프로세스<br>
1. 데이터수집(web-crawling)<br>
2. 감정 labeling / 핵심문장추출 <br>
3. 핵심키워드 추출<br>
4. AI학습하기<br>
<br>

## 프로그램 실행순서<br>
1. crawler 실행: python crawler.py<br>
2. training: python train.py<br>
3. test: python test.py  (미리 수집된 데이터와 학습된 모델로 테스트)<br>
<br>

## Prerequisites<br>
- tensorflow1.15<br>
- konlpy, Komoran<br>
- PyQt5<br>
- selenium<br>
<br>


