---
title: "머신러닝의 기본개념"
excerpt: "모두의 딥러닝 시즌2_tensorflow"
toc: true
toc_sticky: true
categories:
  - ML
tags:
  - ML
last_modified_at: 2022-06-20T08:05:00+09:00
---



> 도커는 알아서 [설치](https://github.com/deeplearningzerotoall/TensorFlow/blob/master/docker_user_guide.md)할 것  
> 모두의 딥러닝 강의

![img](https://deeplearningzerotoall.github.io/season2/assets/video.png) YouTube Lecture: http://bit.ly/2HHrybT


<br>

## Machine Learning

**일종의 프로그램**이다. 우리가 평상시에 알고 있는 일반 프로그램이 입력을 기반으로 어떤 데이터를 보여주는 것 처럼 머신러닝도 입력을 기반으로 어떤 결과 데이터를 보여주는 **하나의 애플리케이션**이다.

그렇다면 어떤 프로그램 구현하고 싶을 때 머신러닝을 활용하면 좋을까?  
어떤 정확하게 <u>모든 프로그램을 조건을 고려 하는 데 제한이 있는 프로그램</u>들이 있다.   
| ex: spam filter, Automatic driving

이런 제한 사항이 있는 프로그램을 다 구현하기에는 너무 힘이 든다.  

특정한 자료에서 발견할 수 있는 어떤 현상을 컴퓨터가 알아서 배우는 건 어떨까? 이것이 바로 머신 러닝

개발자가 일일이 다 정하지 않고 프로그램이 스스로 데이터를 학습해서 능력을 갖는 것이 머신러닝.

<br>

## Learning이란

### Supervised learnining

어떤 정해져있는 **labeled data(training set)**을 가지고 학습하는 러닝.   
예를 들어서 고양이나 개나 이미지 별로 모아서 이를 고양이라고 label을 달아서 학습을 시키는 것이다. 이런 학습을 **supervised learning**이라고 부른다.

많이 다룰 것임  
| image labeling, Email spam filter, Predicting exam scores ...  
밑에서 계속.

### Unsupervised learning

어떤 데이터는 일일이 레이블을 줄 수 없다. 예를 들어 google new는 유사한 뉴스를 모아주는 역할을 하는데 이는 수만 가지의 모든 조건을 다 일일이 label을 달기가 힘들다. Word clustering도 마찬가지이다.

> 교수님 comment: Unsupervised learning를 통해 데이터를 분류를 할 수 있지만 까보기 전에는 이게 TRUE인지 FALSE인지 알 수가 없다. 결함 예측 분야에서는 여기에서 끝나면 힘듦. 결함 예측에서는 [CLAMI](https://scholar.google.co.kr/citations?view_op=view_citation&hl=ko&user=BYm7qHAAAAAJ&citation_for_view=BYm7qHAAAAAJ:epqYDVWIO7EC)는 Unsupervised learning 후 label까지 직접..?

우리는 **supervised learning**에 대해서 좀 깊게 공부 할 것이다.


<br>
## Supervised learning

**Types**

### regression
Final exam score처럼 0~100으로 답이 정해져있는 것, 연속적인 결과값.

### classification

- binary classification
  :pass /fail 와 같이 둘중의 하나를 고르는 것, 분류 하는 것
- multi-label classification
  : Grade를 주고 싶은 시스템을 만들고 싶다. A,B,C,D,F

#### Training data set

머신 러닝이란 하나의 모델에 특징x과 답y(label)이 쓰여있는 데이터 테이블(training data set)을 모델에게 넘겨주어 이를 학습시킴으로써 hypothesis(모델)을 만들어내는데, 이 모델을 활용하여 후에 새롭게 발견된 변수 x값을 입력하면 모델에서   ./예측 결과 값 인 y값을 예측해주는 것.

~~~
~~~

알파고도 여러가지 요소가 있는데, 알파고가 한 일이 사람들이 다룬 바둑알 기본을 학습함.





다음 시간에는 **Linear regression**

---




