<script type="text/javascript" async src="http://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$'], ['\\(','\\)']]}});</script>

<!-- MarkdownTOC autolink="true" bracket="round" markdown_preview="markdown" -->

# Maximum Entropy Inverse Reinforcement Learning (2008) - Review


<p align="right"> 
Geonhee Lee
</p>

------


Author: Brian D. Ziebart, Andrew Maas, J.Andrew Bagnell, and Anind K. Dey School of Computer Science, Carnegie Mellon University, Pittsburgh

Proceedings of the Twenty-Third AAAI Conference on Artificial Intelligence (2008)


------


## Abstract

이 논문에서는 **maximum entropy의 원리에 기반하여 확률론적 접근법**을 개발했다.

논문에서의 접근법은 decision sequence들에 대해 well-define된 **globally normalized distribution**을 제공하고, **동시에 기존 방법과 동일한 성능 보장을 제공한다.**

저자는 내재적으로 noisy하고 완벽하지 않은 수집된 data인 real-world navigation 및 driving behavior들을 모델링된 상황에서 저자의 기술을 개발했다.

**저자의 확률론적 접근법은 부분 경로을 기반으로 경로 및 목적지를 추론하는 강력한 새로운 접근방식 뿐만 아니라 route preference 모델링을 가능하게 한다.**



------



## Introduction

imitation learning의 문제에서의 목표는 agent가 선택할 behavior 및 decision을 예측하는 것을 배우는 것이다. 
(e.g., 사람이 물체를 잡기위한 모션을 어떻게 할것인가? 혹은 운전자가 집에서 직장으로 가기위한 경로는 어떻게 설정하여 갈것인가?)

목적이 있는 순차적 의사 결정 행동을 포착하는 것은 general-purpose statistical machine learning 알고리즘에서 매우 어려울 수 있다; 이러한 문제를 풀기 위해서는 **종종 먼 미래의 action의 결과에 대해 추론**해야만 한다.


(기존아이디어)
-   Imitation learning의 문제에 접근하기 위한 최근 아이디어는 학습된 policy의 공간을 search, planning하거나 더욱 일반적으로 MDP의 솔루션으로 구조화 한다.
-   직관적으로 핵심 개념은 agent는 알려지지 않은 reward 함수(특징이 선형이라 가정)를 최적화하고, **시연되는 행동을 (near)-optimal으로 나타나게하는 reward weight를 찾아야만 한다.**
-   **Imitation learning 문제**는 reward function을 최적화하는 long, coherent한 의사 결정 시퀀스를 "결합"한 search 알고리즘을 가지고 시연행동을 유도한 **reward function의 recovering 작업으로 축소**된다.
(comment) Imitation learning문제는 시연자의 행동을 가지고 보상함수를 복구하는 문제로 볼 수 있으며, 이러한 문제는 search 알고리즘(최적점을 찾는)으로 구조화된 문제를 풀어야 한다.


(저자 제안)
-   저자는 imitation learning에서** uncertainty에 대해 추론하는 완전한 확률적 접근**을 이용.
-   시연된 행동을 reward 함수를 매칭하는 constraint 하에서, decision들의 distribution을 선택할때 **ambiguity를 해결하기 위해 maximum entropy** 원리를 사용.
-   저자는 non-deterministic MDP에 대한 추론을 tractable하게 만들기 위해 추가적으로 가정을 단순화하여 진행.

Distribution의 결과는 다음과 같은 확률적 모델:
1.  행동에 대해 globally normalize.
2.  planning system의 dynamics을 통합하는 conditional random field를 연결하기 위한 확장.
3.  infinite horizon까지 확장.

(comment) 전문가의 시연을 reward 함수로 매칭하기 위해 constraint를 추가하고, max ent 원리를 사용하여 결정에 확률분포를 주고 모호성을 해결.


저자의 연구 성과는 driver의 real-world routing preference을 모델링하는 문제에 의해 동기 부여되었다.
저자는 max ent 접근법을 taxi-cab driving의 수집된  GPS 데이터를 100,000마일 사용하여 route preference 모델링에 적용했다. 여기서 world의 구조(i.e., road network)은 알려져 있고, 이용가능한 action(i.e., road segment를 가로지름)은 road feature(e.g., 속도제한, 도로의 수)을 특징으로 한다.
많은 imitation learning 기술과 현저하게 대조적으로, 저자의 행동의 확률 모델은 hidden variable 기술을 포함하는 다른 확률적 방법과 완벽하게 통합된다. 이것은 부분적 경로에 기반하여 미래의 경로 및 목적지를 자연스럽게 추론할 수 있도록 hidden goal을 가지고 route preference을 확장할 수 있게 한다.


주요 관심사는 시연된 행동(전문가)이 **noise하거나 완벽하지 않은 행동을 하는 경향이 있다는 것이다.**
-   Maximum entropy 접근법은 이러한 **uncertainty를 다루는** principled method를 제공한다.   
  
저자는 (1. margin methods (Ratliff, Bagnell, & Zinkevich 2006)를 포함하는 irl) 및 (2. 이용가능한 행동에서 locally normalize(Ra- machandran & Amir 2007; Neu & Szepesvri 2007))하는 접근법이 행동을 모델링하는 것에 대한 추가적인 장점을 논의한다.


------

## Background

Imitation learning에서 여러 planning space에서 agent의 행동은 agent를 모방하거나 모델링을 시도하는 **learner에 의해 관측**된다.


Agent는 각 state에서 feature들을 그 state를 방문하는 agent의 utility($f_{s_j}$)를 나타내는 state *reward value*로 선형적으로 매핑하는 일부 함수를 최적화하기 위해 시도되어진다고 가정한다. 이 매핑함수는 *reward weight*( $\theta$ )에 의해 매개 변수화 된다.

**trajectory의 reward value**는 간단히 state reward들의 합이거나, 동등하게 path feature count($f_{\zeta}$ = $\sum_{s_j \in \zeta} f_{s_j}$) 에 적용된 reward weight이다. 이것은 trajectory를 따른 state feature의 합이다


$\qquad$$\qquad$        reward($f_{\zeta}$) = $\theta^T f_{\zeta}$ = $\sum_{s_j \in \zeta} \theta^{T} f_{s_j}$ 

### Notation

-   agent's behavior(learner에 의해 관측되는 planning space):
    -   trajectory(or path): $\zeta$
    -   state: $s_i$
    -   action: $a_i$
    -   linearly mapping from features of each state to a state reward value: $f_{s_j}$ $\in \Re^k$
        -   reward value: represent agent's utility for visiting that state
        -   reward weight: $\theta$
        -   reward($f_{\zeta}$): $\theta^T f_{\zeta}$ = $\sum_{s_j \in \zeta} \theta^{T} f_{s_j}$ 
    -   state transition distribution: T
      
-   Demostration of agent:
    -   single trajectories: $\tilde{\zeta}_i$
    -   expected empirical feature count: $\tilde{f}$ = $\frac{1}{m}$ $\sum_i f_{\tilde{\zeta}_i}$
    -   $m$: the number of demostrated trajectories

-   learner:
    -    expected visitation frequencies: $D_{s_i}$
  



### MMP

Agent의 정확한 reward weight을 recovering 하는 것은 ill-posed 문제; Degeneracies(e.g., 모두 0)를 포함하는, 많은 reward 가중치가 시연경로를 최적으로 만든다.

Ratliff, Bagnell, & Zinkevich(2006)은 *structured maximum margin prediction(MMP)* 로 이 문제를 제기했다.
-   Agent와 학습된 policy 사이의 불일치를 직접적으로 측정하여 loss function들의 class를 고려.
-   그 다음, structured margin method를 사용하고 MDP solver에 대한 oracle access만을 요구하여 이 loss의 convex relaxation에 기반한 reward fucntion을 효율적으로 학습한다.
-   그러나, **single reward 함수**가 시연행동을 대체행동(alternative behavior)보다 최적 혹은 상당히 좋게 만들 수 없을 때 mmp는 심각한 결점이 있다.(예를 들어, agent에 의한 시연행동이 불완전하거나 planning 알고리즘이 state-space와 관련된 일부만을 포착(자주 발생)할때, 관측 행동을 완벽히 묘사할 수 없다.)

(comment) 역강화 학습을 통해 학습된 행동(대체 행동?)이 (노이지한)시연된 행동보다 최적일 수가 없다는 점을 말하는 건가? or single reward function의 단점을 얘기하는 것?



### APP

Abbeel & Ng (2004)은 Inverse Reinforcement Learning (IRL) (Ng & Russell 2000)에 기반한 다른 접근을 제공했다.

저자는 관측되는 policy와 learner의 행동간의 feature expectation(Eqn 1)을 매칭하는 전략을 제안했다;
    -  이 매칭은 agent가 특징들이 선형인 reward function을 가진 MDP를 푸는 경우라면, agent와 같이 같은 성능을 성취하는 필요충분조건이라는 것을 증명한다.
      
$\qquad$$\qquad$  $\sum_{Path \space \zeta_i} P(\zeta_i) f_{\zeta_i}$ = $\tilde{f}$ (1)




앞의 IRL 개념과 feature count의 매칭은 ambiguous하다.
-   각 policy는 많은 보상 함수(e.g., all zeros)에대해 optimal할 수 있고 많은 policy들은 같은 feature count들을 유도한다.
-   sub-optimal 행동이 시연될 때, policy의 혼합이 feature count를 매칭 요구하거나, 유사하게 많은 policy의 다른 혼합은 feature matching을 만족한다.

ambiguity를 해결할 방법은 제안되지 않았다.
-   이 논문에서 제시.

</br>
</br>
</br>

------

## Maximum Entropy IRL

-   저자는 principled way로 ambiguity를 다룰 수 있도록 하는 feature count들을 매칭하는 다른 접근법을 이용했고 **single stochastic policy를 도출한다.**
-   저자는 distribution 선택하여 ambiguities를 해결하기 위해 principle of maximum entropy(Jaynes 1957)을 사용했다.
-   이 원리는 feature expectation에 매칭되는 constraint된 행동에 대하여 distribution을 유도.

------

</br>
</br>
</br>

## Deterministic Path Distribution

Policy들에 대한 추론하는 이전 연구와는 달리, 저자는 가능한 행동의 전체 class에 대해서 distribution을 고려한다.

이것은 deterministic MDP(Figure 1a)에 대해 (잠재적으로) 변하는 길이(Figure 1b)의 경로와 상응한다.

**Policy들의 distribution들과 유사하게, 어떤 시연된 행동이 sub-optimal일 때 많은 다른 path의 distribution은 feature count와 일치한다.**

<p align="center"> 
<img src="./figure1.png" width="400" height="230">
</p>


이러한 set 중 하나의 distribution은 path feature에 의해 암시되지 않는 다른 것들에 대해서 몇몇 path에 대한 preference를 나타낼 수 있다.

(comment) path feature는 사진에서 이것이 도로인지 아닌지를 나타내는 것이고, 이러한 정보없이 분포를 이용해도 선호도를 구할 수 있다는 것?

**저자는 feature expectation(eqn 1)을 일치하는 것을 넘어서 어떤 추가적인 선호도를 나타내지 않는 분포를 선택하여 ambiguity를 해결하는 maximum entropy 원리를 사용했다.**
(commnet)모호성 제거하기 위해 분포를 이용한다는데 추가적인 선호도를 나타내지 않는다는 것은 무엇?

Deterministic MDP에 대해 path에 따른 결과 distribution는 reward weight $\theta$ (Eqn 2)에 의해 매개 변수화된다.

이 모델에서, reward들과 동일한 plan들은 동일 확률을 가지며 더 높은 보상의 계획은 기하급수적으로 선호된다.

(comment) plan는 reward와 동일하고, 같은 확률을 가짐 

$\sum_{Path \space \zeta_i} P(\zeta_i) f_{\zeta_i}$ = $\tilde{f}$ (1)
$P$($\zeta_i$ | $\theta$) = $\frac{1}{Z(\theta)}$ $e^{\theta^{T} f_{\zeta_i}}$ = $\frac{1}{Z(\theta)}$ $e^{\sum_{s_j \in \zeta_i} \theta^{T} f_{s_j}}$  (2)



</br>
</br>

**파라미터 가중치(partition function, Z($\theta$))가 주어지면, discounted reward weight를 가진 finite horizon 문제와 infinite horizon 문제에 대해 항상 수렴한다.**

zero-reward absorbing state를 가진 infinite horizon 문제에 대하여, partition function은 모든 state들이 reward가 음수일때조차 실패할 수 있다.

그러나, finite한 step 수에서 absorb된 시연 trajectory가 주어지면, entropy를 최대로 하는 reward weight는 수렴된다.

</br>
</br>
</br>

------


## Non-Deterministic Path Distribution

일반적인 MDP에서, action은 state transition distribution, T에 따른 state(Figure 1c)간의 non-deterministic transition을 생성한다.

이러한 MDP(Figure 1d)에서 path들은 agent의 action 선택과 MDP의 random 결과에 의해 결정된다.

path들에대한 저자의 분포는 randomness를 고려해야만 한다.

(comment) randomness를 고려하는 방법은 expectatin을 사용한다고 알고있다.


저자는 transition distribution(T)을 조건으로 하는 path의 maximum entropy distribution을 사용하고 feature expectation(Eqn 1)을 일치하도록 constraint 시킨다.

$\sum_{Path \space \zeta_i} P(\zeta_i) f_{\zeta_i}$ = $\tilde{f}$   (1)

action 결과공간 $T$, 결과 sample $o$을 고려하고, 모든 action에 대해 다음 state를 지정한다.

MDP는 $o$와 호환하는 경로(즉, 경로 및 $o$의 action 결과가 일치)에 대해 이전 분포(Eqn 2)을 가진 주어진 $o$에 대해 deterministic.

(commnet) o는 최종결과이고, T는 action의 선택집합이다. 이 두개가 같으면 deterministic한거임(랜덤성이 없으므로).

$P$($\zeta_i$ | $\theta$) = $\frac{1}{Z(\theta)}$ $e^{\theta^{T} f_{\zeta_i}}$ = $\frac{1}{Z(\theta)}$ $e^{\sum_{s_j \in \zeta_i} \theta^{T} f_{s_j}}$  (2)

Indicator function($I_{\zeta \in o}$)은 $\zeta$가 $o$와 호환할때 1이고 아닐때는 0이다.

distribution(Eqn 3)을 계산하는 것은 일반적으로 intractable.

**그러나, 만약 transition randomness가 행동에 제한된 영향을 미치고 모든 $o \in T$에 대해 partition function이 상수라면, path에 대한 tractable approximate distribution (Eqn 4)을 얻을 수 있다.**


$P(\zeta | \theta , T)$ = $\sum_{o \in T} P_T (o) \frac{e^{\theta^T f_{\zeta}}}{Z(\theta , o)} I_{\zeta \in o}$ (3)
$\approx$ $\frac{e^{\theta^T f_{\zeta}}}{Z(\theta , T)}$ $\prod_{s_{t+1}, a_t, s_t \in \zeta}$ $P_T (s_{t+1} | a_t, s_t)$   (4)

</br>
</br>
</br>

------

## Stochatic Policies

path에 대한 distribution은 Eqn 4의 partition function이 수렴할 때 stochastic policy(i.e.,각 state의 이용가능한 action에 따른분포)를 제공한다.

action의 확률은 그 action으로 시작하는 모든 path의 예상되는 지수적 보상에의해 가중된다.

$P$(action $a$| $\theta, T)$ $\propto$ $\sum_{\zeta : a \in \zeta_{t=0}}$ $P(\zeta | \theta, T)$(5)


</br>
</br>
</br>

------

## Learning from Demostrated Behavior

**관측된 데이터로부터의 feature constraint되는 path에 대한 distribution의 entropy를 최대화하는 것은 위에서 유도한 maximum entropy(exponential family) distribution에 대한 관측 데이터의 likelihood를 최대화한다는 것을 내포한다.**

(comment) cross entropy가 최소화(KLD-divergence도 최소화), likelihood 최대화. MSE는 최소화

$\theta^*$ = 
argmax$_{\theta} L(\theta)$ = 
argmax$_{\theta} \sum_{examples} log P(\tilde{\zeta} | \theta , T)$ 


**이 함수는 deterministic MDP에 대해 convex이고 optima는 gradient-based optimization을 사용하여 얻어진다.**

gradient는 expected empirical feature count와 learner의 expected feature count간의 차이이고, 이것은 expected visitation frequencies($D_{s_i}$) 관점에서 표현될 수 있다.

$\nabla L(\theta)$ = $\tilde{f}$ - $\sum_{\zeta} P(\zeta | \theta, T) f_{\zeta}$ = $\tilde{f}$ - $\sum_{s_i} D_{s_i} f_{s_i}$   (6)

(comment) trajectory를 state로 변경

**최대 지점에서, feature expectation은 일치함, learner는 agent가 최적화(Abbeel & Ng 2004)를 시도하는 실제 reward weight와 상관없이 agent의 시연된 행동과 동등하게 수행되도록 보장한다.


**실제로, 저자는 모방되어질 agent의 true value가 아닌 feature value의 경험적, 샘플기반 기댓값을 측정한다.**

feature의 크기가 bounded될 수 있다고 가정, standard union 및 Hoeffding bound argument는 샘플 수의 함수로서 feature expectation에서 error에대해 높을 확률 경계를 제공할 수 있다 - 특히, 이러한 경계는 feature의 수에 대해 O(log K) 의존성만 가진다.

>1. 반대로, margin-based 및 locally normalizing model은 feature의 수에 선형적으로 증가한다.

</br>

Dud ́ık&Schapire (2006)는 feature expectation에서 주어진 bound된 uncertainty 결과를 도출하는 maximum entropy 문제가 위에서 서술한 것과 정확하게 같은 *maximum a poseteriori* 문제이다. 그러나 L1-regularizer가 추가되었다(feature expectation에서 uncertainty에 의존하는 regularization의 강점을 가지고)
저자의 실험 섹션에서, online exponentiated gradient descent 알고리즘을 사용하는 것이고 이것은 매우 효율적이며 계수들을 regularizing하는 효과인 $l_1$ 형태를 유도한다.

>2.Stochastic MDP의 경우, 저자는 MDP에 uncerainty으로 인해 sample feature expectation의 분산을 제거하여 finite data의 더 좋은 사용성을 성취할 수 있다. 

Spce는 완벽하지 않은(그리고 non-convex) log-likelihood의 full exposition(정의, 설명)를 허용할 수 없지만, 결과적으로  직관적인 expectation maximization 알고리즘은 초기 feature expectation을 사용하여 maximum-entropy를 fit한 다음 MDP에서 결과 policy를 실행하여 이러한 추정치(estimates)를 향상시킨다.


</br>
</br>
</br>


--------


## Efficient State frequency Calculations

expected state frequencies가 주어지면, 최적화를 위해서 gradient가 쉽게 계산될 수 있다.(Eqn6)

expected state frequencies를 계산하는 가장 명료한 방법은 각 가능한 path를 열거하는 것이다.

불행하게도, MDP time horizon을 가지고 path의 기하급수적인 증가는 enumeration-based approach들이 계산적으로 infeasible하게 만든다.


<p align="center"> 
<img src="./figure2.png" width="400" height="450">
</p>

대신에, 저자의 알고리즘은 RL에서 Conditional Random Field 혹은 value iteration으로 forward-backward 알고리즘과 유사한 기술을 사용해서 효율적으로 expected state occupancy를 계산했다.

**이 알고리즘은 large fixed time horizon을 사용하여 infinite time horizon에 대한 state frequencies를 근사화시킨다.**

-   (Step1) 이것은 재귀적으로 각 가능한 terminal state에서 "back up"
-   그리고 각 action 및 state에서 Eqn4에 대한 partition function을 계산하여 (Step2) 방법을 따라 각 branch와 연관된 확률질량함수를 계산한다.
-   (Step3) 이러한 분기 값은 local action 확률을 계산하고
-   (Step4 and 5) 각 timestep에서 state frequencies로부터 계산될 수 있으며
-   (Step 6) 총 state frequency count를 합한다.
  

> Conditional Random Fields(조건부 무작위장): 
>> 일반적인 분류자(영어: classifier)가 이웃하는 표본을 고려하지 않고 단일 표본의 라벨을 예측하는 반면, 조건부 무작위장은 고려하여 예측한다.
 자연언어로 된 글 또는 생물학적 서열 정보, 그리고 컴퓨터 비전 분야에서의 일련의 데이터에 대한 라벨 예측, 분석하는 데 사용되기도 한다. 
 구체적으로, 조건부 무작위장은 부분구문분석, 개체명 인식, 유전자 검색 등의 응용 분야에 사용될 수 있으며, 이러한 분야에서 은닉 마르코프 모델의 대안이 될 수 있다. 
 컴퓨터 비전 분야에서는 객체 인식, 이미지 분할에 종종 사용

> forward-backward algorithm
>> https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm


</br>
</br>
</br>

------

### Diffrent derivation

-   reward function에 대해서 각 policy가 최적이 될 때, ambiguity를 해결하기 위해 Max Ent 원리 사용하고, 많은 정책들은 IRL 및 feature matching에서 같은 feature count를 유도한다
-   목표: 시연들(expert collected data)에서 log likelihood를 최대화
-   가정: trajectory의 reward는 feature count을 가지고 linear combination으로 표현된다.


이 문제를 풀기 위한 constraint.

-   Feture matching.

$\qquad$$\qquad$        $\sum_{Path \space \zeta_i} P(\zeta_i) f_{\zeta_i}$ = $\tilde{f}$ (1)

(comment) constraint는 확률이다

-   maximise the log likelihood

$\qquad$$\qquad$        $max \sum_{\tau} P(\tau)$ log $P(\tau)$ 

**이 문제를 풀기 위해서 Lagrange multiplier를 사용할 수 있다.**

$\qquad$$\qquad$        $L$ = $\sum_{\tau} P(\tau)$ log $P(\tau)$  - $\lambda$ ($\sum_{\tau} P(\tau) f_{\tau}$ - $\tilde{f}$) - $h$ ($\sum P_{\tau} -1 )$)

(comment) $\tau$ = path or trajectory, $\lambda$, $h$ = lagrange multiplier

$\qquad$$\qquad$        $dL$ = $\sum_{\tau}$ log $P(\tau)$ - $\sum_{\tau}(1)$ - $\lambda$ ($\sum_{\tau} f_{\tau}$) - $h$ $\sum_{\tau}$(1) = 0

(comment) P의 미분값은 1, equaility condition(?): $h$+1 = $\lambda_0$

$\qquad$$\qquad$        $dL$  = =$\sum_{\tau}$ [ log$(P)$ + f($\tau$) + $\lambda_0$ ] = 1

$\qquad$$\qquad$        P($\tau$) = e$^{-\lambda_0 - \lambda f_{\tau}}$ $\Rightarrow$ $\sum_{\tau} P(\tau)$ = 1

$\qquad$$\qquad$        e$^{-\lambda_0}$ = $\frac{1}{\sum_{\tau} e ^{- \lambda f_{\tau}}}$ = partition function(Z = $\sum_{\tau} e ^{- \lambda f_{\tau}}$).

$\qquad$$\qquad$        P($\tau$) = $\frac{1}{Z}$ e $^{- \lambda f_{\tau}}$ = $\frac{1}{\sum_{\tau} e ^{\theta^T f_{\tau}}}$ e $^{\theta^T f_{\tau}}$

(comment) Partition function 유도 완료.

**Maximum entropy 원리 (Jaynes 1957)**:
-   시연된 trajectory의 확률은 trajectory의 reward의 exponential에 비례한다.

$\qquad$$\qquad$        $P(\tau)$ $\propto$ $e^{r(\tau)}$

(comment) 여기서 $r(\tau) = \theta^T f_{\tau}$

그리고 목적은 시연된 trajectory들의 log likelihood를 최대화하는 것이다.

$\qquad$$\qquad$        $\theta ^{*}$ = argmax $_{\theta} L( \theta )$ 

$\qquad$$\qquad$        = argmax $_{\theta } \frac{1}{m}$ 

$\qquad$$\qquad$        $\sum_{\tau_d \in D}$ log P(r( $\tau_d$ ))

(comment) 여기서 $\tau_d$ = expert의 trajectory, 위식에 r()로 대체

$\qquad$$\qquad$        $L(\theta)$ = $\frac{1}{m} \sum_{\tau_d}$ log $\frac{1}{Z}$ e $^{r({\tau_d)}}$ = $\frac{1}{m} \sum_{\tau_d}$ ($r(\tau_d) - log (Z)$)

$\qquad$$\qquad$        $\frac{dL}{d \theta}$ = $\frac{1}{m}$  [ $\sum_{\tau_d}$  $\frac{d r(\tau_d)}{d \theta}$ - $\frac{ 1 \cdot \sum_{\tau} e ^{r(\tau)} }{\sum_{\tau} e^ r(\tau)}$ $\frac{d r(\tau)}{d \theta}$] = $\frac{1}{m}$ [ $\sum_{\tau_d}$  $\frac{d r(\tau_d)}{d \theta}$ - $\sum_{\tau} P(\tau)$ $\frac{d r(\tau)}{d \theta}$ ]

(comment) 위의 식에는 P($\tau$) = $\frac{1}{Z}$ e $^{- \lambda f_{\tau}}$ 대입, 아래의 식은 $r(\tau)$ = $\theta^T$ * $f_{\tau}$ 대입.

$\qquad$$\qquad$        $\frac{dL}{d \theta}$ =  [ $\tilde{f}$ - $\frac{1}{m}$ $\sum_{\tau} P(\tau)(f_{\tau})$ ]

(comment) trajectory를 state로 변환

$\qquad$$\qquad$        $\frac{dL}{d \theta}$ = [ $\tilde{f}$ - $\frac{1}{m}$ $\sum_{\tau} P(\tau)(f_{\tau})$ ] = [ $\tilde{f}$ - $\sum_{s \in \tau}$ $\frac{ P(s | \theta , T)}{m}$ $\cdot$ ($f_{s}$) ]

(comment) state visitation frequencies = $\sum_{s \in \tau}$ $\frac{ P(s | \theta , T)}{m}$ = $D_{s_i}$

$\qquad$$\qquad$        $\frac{dL}{d \theta}$  = $\frac{1}{m}$ $\sum_{s \in \tau_d}$ [ $\frac{d r(s_d)}{d \theta}$ - $\sum_{s \in \tau} P(s | \theta, T)$ $\frac{d r(s)}{d \theta}$ ] $\Rightarrow$ $\bar{f}$ - $\sum_{s \in \tau} D_s f_s$

(comment) 여기서 $D_s$ = $\frac{P(s | \theta, T)}{m}$, $f_s$ = $\frac{d r(s)}{d \theta}$, $\bar{f}$ = expert demonstration, $\sum_{s \in \tau} D_s f_s$ = policy



**Dynamic programming**

주어진 optimal policy $\pi (a,s)$ 와 transition matrix가 주어진 것에 대해 state visitation frequency를 계산하기 위해 DP 사용.
매 시간 t에 state를 방문하는 확률을 선언하기 위해 $\mu$를 사용할 수 있다.

<p align="center"> 
<img src="./figure3.png" width="350" height="200">
</p>

(comment) 여기서 $\mu$는 $D_{s}$(expected visitation frequencies ), $\pi (a,s)$ = P(a | s), P(s) = $\sum_t \mu_t (s)$

**Algorithm**

1.  $\theta$ 초기화, 시연 D 수집
2.  reward r($\tau$)에 대한 최적 정책 $\pi (a, s)$에 대해 풀기
3.  state visitation frequency p(s|$\theta$)에 대해 풀기
4.  gradient $\nabla_{\theta} L$ 계산
5.  $\nabla_{\theta} L$을 사용하여 하나의 gradient step으로 $\theta$ 갱신
6.  step 2 반복

(comment) $\theta$ = reward weight, D= expert 



</br>
</br>
</br>

---------


## Reference

1. [Maximum Entropy Inverse Reinforcement Learning](https://scholar.google.com/scholar_url?url=http://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf&hl=ko&sa=T&oi=gsb-ggp&ct=res&cd=0&d=4466945277776011021&ei=zYoMXPTeOoX3ygSL04HwAg&scisig=AAGBfm0t3Ei88Eq2SYlCauqYm9-uzuVt_g)