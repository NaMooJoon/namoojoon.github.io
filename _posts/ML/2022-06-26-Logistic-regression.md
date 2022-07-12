---
title: "[5] Logistic Regression"
excerpt: "Logistic regression에 대한 기본적인 개념들"
toc: true
toc_sticky: true
categories:
  - ML
tags:
  - ML, Logistic,regression, cost, classification,sigmoid
last_modified_at: 2022-06-26T08:05:00+09:00
---

## Lab 05-1 Logistic Regression

Logistic Regression은 분류 기법 중에 하나로 동그라미, 세모 등 다른 특징을 가지고 있는 것들을 분류하는 데 많이 사용되는 알고리즘. 또한 추후에 공부하게 되는 뉴럴 네트워크의 구조와 딥러닝에 대한 이해 하는 데 기초가 되는 내용

**[목차]**

- What is Logistic Regression?
  - Classification
  - Logistic vs Linear
- how to solve?
  - Hypothesis Representation
  - Sigmoid / Logistic Function
  - Decision Boundary
  - Cost Function
  - Optimizer (Gradient Descent)

> 참고 사이트: Andrew Ng's ML Class 





## Classification

우리가 자격증 시험을 치면 합격과 불합격을 받을 것이다.  
우리가 메일을 받는다면 스팸이거나 스팸 메일이 아니거나 분류할 수 있다.

이처럼 TRUE / FALSE 형태로 나눌 수 있는 분류를 우리는 Binary Classification이라고 부른다. 이 Binary classification은 후에 Multi-class classification의 기본이 되는 내용이다.

Logistic Regression은 binary classification을 구현하기 위한 알고리즘으로 data의 특징에 따라 두가지의 케이스로 구분선을 구별해준다.

```python
x_train = [[1,2],[2,3],[3,4],[4,3],[5,3],[6,2]]
y_train = [[0],[0],[0],[1],[1],[1]] # One Hot
```

## Logistic vs Linear

![image](https://www.researchgate.net/publication/335786324/figure/fig1/AS:802479209971712@1568337361258/Logistic-regression-and-linear-regression.jpg)

### Logistic

- 두가지 케이스를 구분 해주는 구분 선 (Discrete)
- 데이터들은 분류 별로 셀 수 있고, 흩어져있다.
- Discrete

### Linear Regression

- 데이터들이 연속 적이다. (Continous)
- 즉 새로운 데이터가 들어온다고 하더라도, 이에 인근하는 이어지는 데이터를 예측해낼 수 있는 기 법



![image](https://pbs.twimg.com/media/FDPaCRiXoAQKJEE.jpg:large)

## Hypothesis Representation

예를 들어 보자.  
X는 Study hour라고 가정을 하고, Y는 Pass or Fail 이라고 하자.

지금까지 우리는 linear regression을 가지고 데이터를 예측하는 모델을 만들 수 있었는데,   
$H_\theta(X) = (\theta^TX), $  $\theta^T $is weight  
`hypothesis = tf.matmul(X, W) + b # linear W is an [1xn+1] matrix / W(theta) are parameters`
이런 경우에는 Linear Regression을 가지고는 데이터를 분류할 수 없다. 왜냐하면 결과 값이 0과 1로 구분이 되어 있기 때문이다.



Study hour가 어느 임계점에서 더 늘어나는 순간 그 이후로 부터는 pass라고 구분을 discrete하게 할 수 있어야 한다.
이를 구현하기 위한 과정은 다음과 같다.  

1. 첫번째는 data set에 대한 오차를 줄일 수 있는 linear function을 구해 낸다. $(\theta^TX)$
2. 1에서 구한 Linear function을 Logistic function을 통해서 결과 값의 범위를 $(0\le y\le 1)$ 조정한다. $g(\theta^TX)$
3. $(y \ge 0.5)$ 를 기준으로 Decision Boundary를 구한다.



## Sigmoid / Logistic Function

**$g(z)$ 에 대하여**

> **Sigmoid function**
>
> S자형 곡선 또는 sigmoid 곡선을 갖는 수학 함수이다. Logistic regression 모델을 구하기 위한 함수로 쓰인다.
> $d(X) = \frac{1}{(1+e^{-X})}$

$g(z)$ => $z$ is a real number => $g(z) = e^z/(e^z+1) = 1/(1+e^{-z})$

> **Euler's numbers**
>
> $e := \displaystyle \lim_{n \to \infty} (1 + \frac{1}{n})^n$
>
> $e = 2.71828182845 ...$


<img src="https://www.mathsisfun.com/sets/images/function-exponential-e.svg" style="float:left; margin-right: 10px;"> 

따라서 $z$값이 커지면 커질 수록 $g(z)$ 값은 1에 수렴하게 될 것이고,  

$z$값이 작아지면 작아질 수록 $g(z)$ 값은 0에 수렴하게 된다.  
<div style="clear:both;"></div>

결과 적을 $g(z)$ 의 그래프는 **sigmoid function**으로 다음과 같이 그려지게 된다.

 <img align="left" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png" width="300"/>
 <div style="clear:both"></div>



## Simoid function

```python
hypothesis = tf.sigmoid(z) # z=tf.matmul(X, W) + b
hypothesis = tf.div(1., 1. + tf.exp(z))
```

텐서플로우에서는 다음과 같이 두 방법으로 sigmoid 함수를 구할 수 있다.



## Decision Boundary

그리고 나서 0.5를 기준으로 0과 1로 구분 할 수가 있다.

```python
predicted = tf.cast(hypothesis > 0.5, dtype=tf.int32)
```



## Cost Function

지금까지 Logistic Regression의 모델 구조에 대해서 설명을 하였는 데, 여기서 부터는 학습을 하기 위한 **Cost function**에 대해서 설명할 것이다. 

먼저 cost function에 대해서 설명하기 전에 cost function의 식을 한번 보자. 



> The cost function to fit the parameters($\theta$)
>
> - $h_\theta(x) = y,\space then\space Cost=0$ 
>
> - $cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))$



**Cost function**은 위와 같이 나타낼 수 있는 데, 이를 **tensorflow**에서는 아래와 같이 호출 가능하다.

```python
def loss_fn(hypothesis, labels):
	cost = -tf.reduce_mean(label * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
  return cost 
```

  <br>

그렇다면 왜 $cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))$ 이런 수식이 나왔느냐.

![image](https://t1.daumcdn.net/cfile/tistory/998F6E4F5A65F1D321)

이에 대한 답은 왼쪽의 cost function을 오른쪽 cost function처럼 만들기 위해서이다.  
즉, non-convex함수를 convex함수로 변환하기 위해서이다.

왼쪽의 식을 수식으로 적으면 다음과 같다.

>  $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m}(h_\theta(x_i) - y_i)^2$

이는 우리가 처음에 Linear regression에서 cost 값을 구하는 식과 같다. 하지만 이런식으로 값을 구하게 되면, 함수가 non-convex함수 꼴로 그려지게 된다. 이는 **Gradient descent algorithm**에서 문제를 일으키게 된다.

따라서 우리는 오른쪽의 함수와 같은 모양으로 cost 함수를 변환하고 싶다. 이를 위해서 우리는 $log$함수를 이용하는 것이다.

> $log(x)$
>
> ![image](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Logx.svg/319px-Logx.svg.png?20140120150228)

> $Cost(h_\theta(x), y)= \begin{cases}
> 	-log(h_\theta(x),\space\space if\space y= 1\\ 
> \\
>  	-log(1-h_\theta(x)),\space\space if \space y=0 
> \end{cases}$

$log$함수를 이용하여 Cost를 위와 같이 재정의를 한다면, 우리는 convex한 함수를 도출해낼 수 있는 것이다.

이를 다시 정리하면  $cost(h_\theta(x),y) = -ylog(h_\theta(x)) - (1-y)log(1-h_\theta(x))$으로 나타낼 수 있다.



### Optimization

우리는 cost function을 convex하도록 만들었다. 따라서 우리는 cost의 최솟값을 찾는 최적화를 Gradient descent alogorithm을 이용해서 해결할 수 가 있다.

> Repeat   ${\theta_j := \theta_j - \alpha\frac{\partial}{\partial \theta_j}J(\theta)}$

[Tensorflow Code]

```python
def grad(hypothesis, labels):
  with tf.GradientTape() as tape:
    loss_value = loss_fn(hypothesis, labels)
  return tape.gradient(loss_value, [W,b])
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
```

