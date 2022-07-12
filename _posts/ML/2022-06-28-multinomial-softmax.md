---
title: "[6] Multinomial Classification"
excerpt: "SoftMax Regression"
toc: true
toc_sticky: true
categories:
  - ML
tags:
  - ML, multinomial,classification, cost, softmax, entropy
last_modified_at: 2022-06-26T08:05:00+09:00
---


## Multinomial이란?

지난 시간에 배웠던 **Simple logistic regression**은 **binomial** 문제로 예측하는 결과 값이 True/False 형태로 이분법적인 형태를 나타냈다. **Multinomial**은 binomial의 확장 버전으로 예측하는 결과 값이 **multi-classes**가 될 수 있을 때 사용한다. 예를 들면 쉽게 이해가 되는 데, 대학교 강의 성적 결과를 생각해보자. 

만일 어떤 특정 강의를 듣고 성적 결과를 예측하는 모델을 만든 다고 가정을 할 때에 이 강의가 Pass/Fail 과목이라고 생각을 해보자. 그렇다면 우리가 예측하고 싶어하는 결과값은 이분법적으로 binomial classification을 적용해야한다.

만일 예측하고자 하는 강의가 Grade로 성적을 매긴다면? 그렇다면 우리가 예측하고 싶어하는 결과는 A, B, C, D, F와 같이 여러개의 결과가 있을 것이다. 이를 우리는 **multinomial**이라 부르며 이를 분류하는 것을 **multinomial classification**이라고 한다. 이번 시간에는 multinomial classification, 그 중에서도 가장 유명한 **Softmax regression**에 대해서 배울 것이다.



## Multinomial classification

**Multinomial classification**은 <u>Binomial classification의 확장판</u>으로 생각할 수 있다. 간단하게 예측하고자 하는 성적을 A, B, C 이렇게 세가지 경우가 있다고 생각해보자. 이런 경우에 binomial 관점에서, A인지 아닌지, B인지 아닌지, C인지 아닌지 binomial classification 방식식으로 구분을 세번 할 수 가 있다. 그러면 다음과 같이 표현할 수 있게 된다. 강의에 나오는 예시 그래프는 다음과 같다.

![image-20220704134501026](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220704134501026.png)

각 예측하고자 하는 결과값을 각각 우리가 배웠던 **binomial**로 각각의 A, B, C의 Hypothesis를 구한 그래프 모습이다. 이를 0과 1로 표현을 하기 위해서 **logistic regression**을 사용해보자.

![image-20220704134636150](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220704134636150.png)

각 결과값의 vector을 matrix로 표현하면 다음과 같다.

![image-20220704134716322](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220704134716322.png)

![image-20220704134803746](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220704134803746.png)

각각의 결과값 $Y_A, Y_B, Y_C$을 **Sigmoid function**을 거쳐 [0,1]의 범위로 표현 할 수 있다.  
예를 들어서, $Y_A=2.0, Y_B=1.0, Y_C=0.1$로 결과값이 나왔다고 하자. 이를 Sigmoid 함수를 적용해 각각 0.87, 0.35, 0.05의 확률 값을 갖게 된다면, 그렇다면 우리는 $Y_A$ 예측 값이 가장 큰 값이기 때문에, 해당 입력 값에 대하여 분류는 A라고 예상할 수 있게 된다.

그런데 여기서 각 요소들의 전체 확률의 합계가 1이 되도록 하여 전체 결과 선택지에 걸친 확률로 표현할 수 있다면 좀 더 직관적이면서 수학적으로 활용하기 편하지 않을까? 예를 들어서 $Y_A$의 확률이 0.7이고, $Y_B$가 0.25, $Y_C$가 0.05로 세 확률의 총합이 1이 되도록 예측 값을 얻을 수 없을까? 이렇게 표현한다면, 확률의 총합이 1이므로, 어떤 분류에 속할 확률이 가장 높을지 쉽게 인지할 수 있다. 이럴 때 사용할 수 있는 것이 **Softmax Regression**이다.







## Soft max regresssion

정리하자면, 소프트맥스는 세 개 이상으로 분류해야하는 **multinomial classification**에서 사용하는 함수이며, 분류할 결과값가 n개라고 할 때, n차원의 벡터를 입력받아서, 이를 각 **결과값에 속할 확률을 추정**한다.

### Soft max function 

$$
Y_k = \frac{e^{a_k}}{\sum_{i=1}^{n}e^{a_i}}
$$

- $Y_k$는 각 결과(출력층, 클래스)에 대한 확률

- $n$은 결과값의 개수(출력층의 뉴런 수, 총 클래스의 수), $k$는 k번째 클래스

- 만일, 총 결과의 개수가 3개라고 한다면 다음과 같다.
  $$
  softmax(z)=[\frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_2}},
              \frac{e^{z_2}}{e^{z_1}+e^{z_2}+e^{z_2}},
              \frac{e^{z_3}}{e^{z_1}+e^{z_2}+e^{z_2}},]
              =[p_1,p_2,p_3]
  $$
  위의 식을 보면, softmax function은 "k번째 결과일 확률 / 전체 확률"로 간단하다는 것을 알 수 있다.



다시 위에서 예를 들었던 것으로 돌아오게 된다면 결과값으로 나오게 된 $Y_A=2.0, Y_B=1.0, Y_C=0.1$를 **softmax**를 통해 다음과 같이 표현할 수 있게 된다.

![image-20220704145259848](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220704145259848.png)



실제로는 하나의 답을 추출해내기 위해서 '**One-hot encoding**'기법으로 하나의 데이터만 1.0값으로 세팅하고 이외의 값들은 0.0으로 처리한다. 이를 구현하는 방법은 매우 쉬울 것이다.

![image-20220705131408633](../../assets/images/posts/다섯번째 모두의 딥러닝//image-20220705131408633.png)

여기까지 **hypothesis** 설정 단계이다. 이렇게 하면 우리의 모델, 우리의 hypothesis를 확률의 표현으로 만들어 낸 것인데(이는 후에 Cross Entropy와 연결되어 수식으로 표현하기 용이함), 문제는 이에 대한 Cost function은 어떻게 설계하는 가이다. 



## Cross Entropy

### Entropy란

Entropy는 불확실성의 척도를 말한다. 즉, 엔트로피가 높다는 것은 각 사건의 확률이 낮다 예측의 불확실성이 높다는 것을 의미하고, 엔트로피가 낮다는 것은 특정 사건의 확률이 높기에 예측의 불확실성이 낮다라는 것을 의미한다.

예시를 들자면, 동전을 튕긴 다음 잡았을 때 앞면이 나올지 뒷면이 나올지 기대하는 상황과, 100개의 공이 있을 때, 노란색 공이 25개, 파란색 공이 25개, 빨간색 공이 25, 초록색 공이 25개 있다면, 랜덤으로 하나의 공을 골랐을 때 그 공이 어떤 색깔의 공일지 맞추는 상황은 모두 Entropy값이 매우 높다. 왜냐하면 어떤 기대값이 나올 지 불확실하기 때문이다.

만약 노란색 공이 70개 있고, 파란색 빨간색 초록색 공이 각각 10개씩 있다면, 노란색의 공이 뽑힐 확률이 높기 때문에 이로 인해 확실성이 높아지게 되고, Entropy값은 이전 상황에 비해서 낮아지게 된다.

정보 이론의 아버지라 불리는 Shannon은 Entropy 값을 구하는 식을 다음과 같이 정의를 내렸다.

> $Entropy=-\sum_{i=1}^{m}p_ilog_2(p_i)$
>
> 여기서 $p_i = \frac{freq(C_i,S)}{|S|}$

여기서 $p_i$를 도출해내는 함수를 보면 우리가 위에서 softmax함수를 쓸 때 사용되는 방법이라는 것을 알 수 있다.

따라서, 우리는 이 Entropy값을 구하는 공식을 이용해서 cost function을 도출해낼 수 있다.

>  컴퓨터공학의 근간이 되는 정보이론에서는 정보량이란 '놀람의 정도'를 의미하는 데, 새롭고 특이한 정보일 수록 사람들로 하여금 놀람을 일으키는 정도가 크며 정보량이 많고, 흔한 정보일 수록 사람들의 놀라움이 작고 정보량 또한 작다라고 표현할 수 있다. 그 이유는 Huffman code 기법을 생각해보면 알 수 있을 것이다. 흔한 정보일 수록 자주 사용되기 때문에 bit로 표현하는 bit length를 작게 만듦으로써 전달되는 정보를 줄일 수 있고, 흔하지 않은 정보일 수록 가끔씩만 사용되기 때문에 앞에서 bit length를 작게 표현한 tradeoff로 bit를 여기서는 크게 표현하게 되기 때문이다.





### Cross-Entropy Function

1. $cost=-\sum_{k=1}^ky_ilog(p_i)$

   - $k$는 class(결과)의 개수

   - $y_i$는 실제 One-Hot 벡터의 j번째 원소

   - $p_i$는 샘플 데이터가 j번째 클래스일 확률

2.  $cost=-\sum_{k=1}^ky_ilog(p_i)$를   
   $D(\hat{Y_i},Y_i)$로도 표현 가능함.
3. 만일, N개의 traininig set이 있을 때는  
   $cost=-\frac{1}{N}\sum_{i=1}^{n}\sum_{j=1}^{k}y_j^ilog(p_j^i)$ 
