---
title: "[3] Multi-variable Linear Regression"
excerpt: "변수가 여러개일 때 linear regression 표현"
toc: true
toc_sticky: true
categories:
  - ML
tags:
  - ML, cost, multivariable, matrix
last_modified_at: 2022-06-21T08:05:00+09:00
---

## Multi-variable linear regression

### 기존의 함수

- Hypothesis				$H(x) = Wx + b$
- Cost function            $cost(W) = {1\over m} \sum_{i=1}^{m}(Wx_i - y_i)^2$
- Gradient descent   $W := W - a$



### EX
 일차퀴즈, 이차퀴즈, 중간고사 점수를 알아냄으로써 기말고사 점수를 예측하는 상황

- 변수가 하나일때 : $H(x) = Wx + b$

- 변수가 여러개일때: $H(x_1, x_2, x_3) = w_1x_1 + w_2x_2 + w_3x_3 + b$



이렇게 변수가 많아진다면, 일일히 다 써주기 힘들기 때문에 ***Matrix***를 사용하는 것이다.  
이 문제를 해결하기 위해서 우리는 ***Matrix multiplication***을 활용하는 것이다.

$H(X) = XW$ ,, 

X를 앞에 오고 W가 뒤에 오는 이유는 행과 열을 계산하기 때문이다.

<br>
---



>## 선형대수학 기본 개념
>
>선형대수는 머신러닝과 딥러닝의 계산을 효율적으로 풀수 있도록 도움을 주기에 알아두어야 모델이 어떻게 만들어 진 것인지 이해 할 수 있다.
>
>
>
>### 스칼라
>
>수학에서 스칼라는 벡터의 한 요소로 언급된다. 스칼라는 실숫값이고 벡터 공간을 정의하는 데, 쉽게 얘기해서 크기만 있고 방향이 없는 물리량이라고 생각하면 된다. 컴퓨터에서는 스칼라가 변수(variable)와 동의어이다. 어떠한 상징적인 이름과 짝을 이루는 저장공간을 뜻하며 이 공간에는 값(value)라는 미지수가 저장되어있다.
>
>
>
>### 벡터
>
>벡터는 양의 정수 n이 있을 때 튜플, 요소 또는 스칼라라고 불리는 숫자 n개로 이루어진 정렬된 set 또는 array이다. 예시를 들어서 보이면 이해가 될 것이다.
>
>벡터의 수학적인 표현법은 다음과 같다.
>
>$x = \begin{bmatrix} x_1 \\   x_2 \\   x_3 \\   ... \\   x_n \end{bmatrix}$
>
>또는 다음과 같이 표현한다.
>
>$x = \begin{bmatrix} x_1 , x_2 , x_3 , ... , x_n \end{bmatrix}$
>
>
>
>### 행렬
>
>Matrix라고 부르는데, 같은 차원(같은 열의 수)을 가지는 벡터들의 그룹이다. 즉, 행렬은 행과 열을 가지는 2차원 배열이다.
>
>
>
>### 텐서
>
>Tensor는 가장 근본적인 수준에서의 다차원 배열으로 벡터보다 더 일반적인 수학적 구조라고 생각하면 된다. 벡터는 텐서의 부분집합이다. 텐서의 차원의 수를 rank라고 하는 데, 스칼라는 rank 0, 벡터는 rank 1, 행렬은 rank 2, 그리고 rank 3 이상은 모두 텐서라고 부른다.
>
>
>
>### 초평면
>
>Hyperplane라고 부른다. 기하학에서 초평면이란 주변 공간보다 한 차원 작은 부분 공간을 말한다. 즉, 3차원 공간에서 초평면은 2차원이고, 2차원에서 초평명은 1차원이다.
>
>초평면은 n차원의 공간을 '부분'으로 나누는 수학적 구조이므로 분류와 같은 문제에서 유용하게 사용된다.
>
>
>
>### 관련 연산
>
>- #### 내적 (dot product)
>
>		'스칼라 곱' 또는 '점 곱'이라고도 불린다. 내적은 유클리드 공간의 두 벡터로부터 실수 스칼라를 얻는 연산이다.
>
>  	$a = (a_1,a_2,...,a_n), \space b = (b_1,b_2,...,b_n)$ 이라고 한다면,
>
>  	$a\cdot b = a_1b_1+a_2b_2+...+a_nb_n$ 이다.
>
>  	$a\cdot b = a^Tb$
>
>
>- #### 성분곱 (element-wise product)
>
>  	두 벡터의 각 요소를 곱하여 같은 길이의 벡터 하나를 생성
>
>
>- #### 외적 (outer product)
>
>  	두 입력 벡터의 '텐서 곱'이다.열 벡터의 각 요소를 행 벡터의 모든 요소와 곱하여 결과 벡터의 새로운 행으로 만든다.