# 과일 가격 예측

## 1. 서론

### 주제

과일 전국도소매시장의 거래 데이터를 수집하여 시간에 따른 과일 가격 추세를 분석하고, 가격을 예측하기 위해 다양한 모델을 활용한 데이터 분석 프로젝트

**1.1 분석 배경 및 목표**

최근 여러 기사에서 과일 및 채소류의 가격 급등 소식을 접했고, 지난해 생산량이 급감한 사과는 계속해서 가격이 상승하여 '금사과'라는 별명이 붙었다는 내용을 보았다. 이러한 가격 급등 현상이 단순히 거래량 감소에 따른 것인지, 아니면 계절적 요인에 의해 거래량과 가격이 함께 변동하는 것인지 궁금증이 생겼다.

이에 따라, 시간적 가격 변동을 체계적으로 분석하여 과일 가격이 어떻게 변화하는지, 그리고 이러한 변화의 주요 요인이 거래량 감소인지, 아니면 계절적 요인에 의한 변동인지 확인해보고자 이 프로젝트를 시작했다. 이를 통해 가격 예측 모델을 구축하여, 향후 발생할 수 있는 가격 변동을 미리 예측하고 분석하는 것이 이번 프로젝트의 주요 목표다.

**1.2 데이터**

- 데이터 출처: 전국도매시장 경락정보
- 데이터 설명:

| **데이터 기간** | 2020-01-01 ~ 2023-12-31 |
| --- | --- |
| **과일 품목** | **총 16가지** 사과, 참외, 귤, 바나나, 복숭아, 키위, 자두, 파인애플, 포도, 망고, 체리, 레몬, 블루베리, 수박 ,딸기, 메론 |

![dataframe](https://github.com/user-attachments/assets/3537b7a8-6c3f-4f84-bc88-3069791d0bdf)

<br/>

## 2. 데이터 전처리

**2.1 데이터 전처리 과정** 

- **fruits**

| **순서** | **COL** | **변경 사항** |
| --- | --- | --- |
| SALEDATE | 경락 일자 |  |
| WHSAL_NM  | 도매 시장 |  |
| CMP_NM | 법인 |  |
| PUM_NM | 품목 |  |
|  KIND_NM | 품종 |  |
| DAN_NM | 단위 | **•** 거래 단위 ⇒  **kg/g/ton(M/T)**  추출 |
| POJ_NM | 포장 | • 거래 단위 ⇒ **상자/팔레트**  추출 |
| ~~SIZE_NM~~ | ~~크기~~ | • 컬럼 삭제 |
| LV_NM | 등급 |  |
| SAN_NM | 산지 | • **광역시도+시군구**<br/>&ensp;◦ (광역시도 = 시군구) ⇒ 광역시도<br/>•  ‘-’ 제외 |
| DANQ   | 단위 중량 | • 거래 단위 ⇒ **숫자**  추출<br/>‼️  **0 인 경우 제외** |
| QTY   | 물량 | • QTY(물량) = (**총 거래 금액** / 단가)<br/>&ensp;◦ TOT_AMT/COST=QTY |
| COST   | 단가 | • COST(단가) = **(총 거래 금액 / 총 거래 물량 )** * 단위중량<br/>&ensp;◦ (TOT_AMT/TOT_QTY)*DANQ=COST<br/>‼️  **0 or Null인 경우 제외** |
| ~~MEAN_COST~~ | ~~평균 가격~~ | • 컬럼 삭제<br/>‼️  **0 or Null인 경우 제외** |
| TOT_QTY   | 총 물량 | • 음수로 집계된 값은 거래 취소 내역<br/>‼️  **0 or Null인 경우 제외** |
| TOT_AMT   | 총 금액 | ‼️  **0 or Null인 경우 제외** |

- **train set**

| COL |  |
| --- | --- |
| **date** | 일자 |
| **요일** | 요일 |
| **품목_거래량(kg)** | 해당 품목의 일자 별 거래량  |
| **품목_가격(원/kg)** | 해당 품목의 일자 별 kg당 가격  |
| **품목 가격 산출 방식** | 품목 또는 품종의 총 거래금액/총 거래량 (※취소된 거래내역(=TOT_QTY<0) 제외) |

<br/>

**2.2 Feature Engineering**

![Feature Engineering Graph](https://github.com/user-attachments/assets/7327b0fa-0a12-4ca5-bf2d-e74a71c1a65c)

![Feature Engineering](https://github.com/user-attachments/assets/19215d59-b0f3-4eba-a51c-0e4ec5acbc86)

<br/>

## 3. EDA

- **품목별 가격 / 거래량 변동**

![EDA 가격거래량](https://github.com/user-attachments/assets/eaf12cfd-2f9b-4dc9-bbad-f96a741eaa90)

- **품목별 가격 분포**

![EDA 가격분포](https://github.com/user-attachments/assets/3af8e971-c574-4957-a986-1417bb9788b7)

<br/>

## 4. 모델링

**4.1 RandomForest**

![RandomForest Score](https://github.com/user-attachments/assets/de7de627-e57b-4ca6-b8be-07dc69417e4b)

**4.2 XGBoost**

![XGBoost Score](https://github.com/user-attachments/assets/25b6392d-e633-4af7-8402-e3c4d7634da0)

**4.3 LSTM**
```
timesteps = 7
x_lstm, y_lstm = create_sequences(x_scaled, y_scaled, timesteps)

x_train, x_test, y_train, y_test = train_test_split(x_lstm, y_lstm, test_size=0.25, random_state=42)

batch_size = 32
epochs = 150
dropout_rate = 0.1
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(dropout_rate))
model.add(LSTM(50))
model.add(Dropout(dropout_rate))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[early_stopping], verbose=2)
```

![LSTM Score](https://github.com/user-attachments/assets/8d6ae16c-272e-470c-b8f0-d361123f2648)

**4.4 결과 분석** 

모델 성능 평가를 위해 train과 test 데이터를 분리하여 1주 후의 사과 가격을 예측했다.

Random Forest와 XGBoost 모델에서는 train 데이터와 test 데이터 간의 정확도 차이가 크게 나타나 과적합이 발생했다고 해석했다. 훈련 데이터는 잘 예측했지만, 테스트 데이터에 대한 예측력은 상대적으로 떨어졌다.

Random Forest와 XGBoost의 정확도는 각각 0.83과 0.83이었으나, LSTM 모델은 train과 test 간 과적합이 적게 발생하였고, 정확도는 0.89로 Random Forest와 XGBoost보다 더 높았다. 

또한 LSTM 모델은 시계열 데이터의 특성을 잘 반영하여 가격 변동의 패턴을 포착하는 데 적합했기 때문에 최종 모델로 선정하였다.

세 모델 중 **LSTM**이 가장 높은 정확도를 보여주었고, 과적합 문제도 적었기 때문에 가격 예측을 위한 최종 모델로 LSTM을 선택했다. 

<br/>

## 5. 결론

**5.1 한계점** 

- **피처 부족**
    
    사용한 피처들은 주로 품목의 거래량, 가격 그리고 요일 등 기본적인 정보에만 기반했다. 하지만 가격 변동에는 기상이나 경제 지표 같은 외부요인들도 큰 영향을 줄 수 있어, 이런 외부 요인을 피처로 추가하면 모델 성능을 더 높일 수 있을 거라 생각한다. 더 정확한 예측을 할 수 있는 방법을 모색할 필요가 있다고 생각한다.
    
- **산지와 날씨 정보를 활용한 예측 어려움**
    
    산지 정보를 활용해 날씨 데이터를 결합하여 feature를 추가하려 했으나, train/test 데이터 생성 시 산지별로 구분된 값이 아닌 국내와 수입의 평균값과 총합만 존재하여 산지별 날씨 데이터를 적용하기 어려웠다. 전국 날씨 데이터를 활용하려 했지만, 이를 찾지 못해 산지 정보를 사용한 예측이 어려워졌다. 가격 예측에서 산지 정보는 중요한 변수 중 하나이다. 그러나 기상 데이터를 포함하려던 시도는 데이터 수집의 어려움과 시간적 제약으로 최종 모델에 반영되지 못했다. 향후에는 날씨 데이터를 포함한 산지별 가격 예측 모델을 개발해 성능을 개선할 수 있을 것이라고 생각한다.
    
- **명절 및 공휴일 전처리 어려움**
    
    명절과 공휴일은 농산물 가격에 큰 영향을 미치며, 특히 가격이 급등하는 시점이다. 하지만 매년 명절과 공휴일 날짜가 달라지기 때문에 이를 전처리하는 데 어려움이 있었다. 향후 명절과 공휴일 데이터를 보다 체계적으로 수집하고, 이를 예측 모델에 반영할 수 있는 자동화된 방법을 구축하면 더 정확한 예측이 가능할 것이라 생각한다.
    

**5.2 향후 방안**

- 외부요인, 산지 및 기상 데이터를 포함한 지역별 맞춤 예측 시스템 개발
- 명절 및 공휴일 효과를 반영할 수 있는 전처리 방법 연구 및 적용
