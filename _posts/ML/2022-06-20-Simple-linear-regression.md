---
title: "Simple Linear Regression"
excerpt: "모두의 딥러닝 시즌2_tensorflow"
toc: true
toc_sticky: true
categories:
  - ML
tags:
  - ML
last_modified_at: 2022-06-20T08:08:00+09:00
---


## Simple Linear Regression(단순 선형)

매우 간단해 보이지만, **머신러닝의 핵심 내용**을 관통하고 있다.


### Regression

우리말로 번역하면 ~~후퇴, 퇴보, 되돌아간다.~~ 라는 의미인데 이런 말은 오해가 될 수 있다.

regression은 긴말을 짧게 줄인 말로 본 뜻은**"Regression toward the mean**" 전체 평균으로 되돌아간다는 의미를 뜻한다. 이것을 줄여서 regression이라고 한다.



#### Linear Regression

: 한마디로 정의하면 데이터를 잘 대변하는 직선의 방적식을 찾는 것

기울기와 y 절편을 구하는 것.

선형회기를 잘 표현하는 사례

| x    | y    |
| ---- | ---- |
| 1    | 3    |
| 2    | 4    |
| 3    | 5    |

#### Hypothesis

$H(x) = Wx + b$

> $H(x)$는 Hypothesis function

가설이 될 수 있는 $W$와 $b$ 에 대하여 여러개의 후보를 구한다.

이때, $H(x) - y(실제데이터)$를 **error**라고 한다.

우리는 이러한 error의 제곱의 합을 **cost**라고 부르는 데

우리는 이러한 <u>cost를 최소화</u> 하는 것이 목표이다.



#### Cost

$cost(W) = {1\over m}\sum_{i=1}^{m}(Wx_i - y_i)^2$

$cost(W,b) = {1\over m} \sum_{i=1}^{m}(H(x_i) - y_i)^2$


우리의 목적은 ***minimize $cost(W,b)$***

> W: weight, b:bias 

> *생각해 볼 것
>
> Q. 각각 weight와 bias가 의미하는 것이 무엇일까?  
> 왜 $cost(W)$와 $cost(W,b)$가 따로 있을까?


---

