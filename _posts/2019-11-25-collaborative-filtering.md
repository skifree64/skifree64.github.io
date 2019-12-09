---
layout: post
title:  "협업 필터링 추천 시스템 (Collaborative Filtering Recommendation System)"
author: 유현식, 임정빈, 정연경, 이준희
date:   2019-11-25
categories: Machine_Learning
tags: recommendation_system

---


[TOC]




# Members

**유현식, 컴퓨터전공**

**임정빈, 원자력공학과**

**정연경, 원자력공학과**

**이준희, 기계공학부**




# 1. Introduction

​	영상을 재생할 수 있는 기기들이 많이 보급되면서 영상 콘텐츠에 대한 수요는 지속적으로 늘어나고 있으며, 이에 따라 수많은 양의 영상 콘텐츠가 실시간으로 공급되고 있다. 하지만 이러한 과잉 공급 때문에, 사용자들은 자신에게 적합한 콘텐츠를 찾는데 어려움을 겪고 있다. 일부 사람들은 이전에 상영했던 영화 중 자신에게 적합한 영화를 찾기 위해 영화 정보 제공 TV 프로그램을 이용하거나 포털 사이트의 지식정보 커뮤니티를 이용하고 있다. 하지만 이러한 소통 방식들은 사용자의 취향이나 의중을 완전히 반영할 수 없고, 추천의 범위가 개인의 지식 내부로 한정되는 단점이 있기 때문에, 적절한 추천이 이루어지기 힘든 경우가 있다. 

​	위와 같은 문제를 해결하기 하여 기업들은 추천시스템을 사용하여 사용자를 확보하고 있으며, 추천시스템은 사용자들에게 그들이 관심 있고 좋아할 만한 아이템을 추천해 주어 원하는 아이템을 쉽고 빠르게 찾을 수 있도록 돕는다. 이 시스템은 온라인 뉴스, 영화, 다양한 형태의 web resource들을 추천하며 많은 e-commerce 사이트에서 사용되고 있다. 추천 시스템이 이러한 서비스에 기여하는 바는 어마어마 하다. Netflix는 대여되는 영화의 66%, Google News는 38% 이상, Amazon은 판매의 35% 가 추천으로부터 발생하며 우리나라에서도 영상 스트리밍 사이트가 등장하면서 영화 추천 시스템이 활성화되고 있다.

​	이번 프로젝트에서는 협업 필터링에 대해서 소개하고 직접 사용자 기반 협업 필터링을 통해 영화를 추천하는 프로그램을 만들 예정이다. 여러 기법을 통해 실험해보고, 그에 따라 프로그램의 영화 평점 예측 성능이 어떻게 달라지는지 살펴보려 한다.




# 2. Datasets

**데이터 셋의 내용**

우리가 사용할 데이터 셋은 ‘MovieLens Data’이다. [Grouplens](http://www.grouplens.org/node/12)라는 미네소타 대학의 컴퓨터과의 연구실에서 수집한 추천 알고리즘을 위한 영화 데이터이다. 여기에는 데이터의 양에 따라 데이터셋이 다양하게 있는데, 그 중에 100K DataSet을 사용할 것이다

[100K]: http://www.grouplens.org/system/files/ml-100k.zip

이 데이터 셋은 총20가지의 데이터로 이루어져 있다. 각 데이터의 내용은 다음의 표와 같다.

| 파일명                     | 데이터 내용                                                  |
| -------------------------- | ------------------------------------------------------------ |
| u.data                     | 943명의 사용자가 1682개의  영화에 남긴 100000개의 평점 데이터. 무작위 배열 |
| u.info                     | u.data의 사용자의 수, 영화의  수, 평점의 수                  |
| u.item                     | 이 데이터에서 사용된 영화의 정보                             |
| u.genre                    | 장르와 각 장르의 코드                                        |
| u.user                     | 유저의 ID, 나이, 성별, 직업, 주소                            |
| u.occupation               | 직업의 종류                                                  |
| u.base, u.test (1~5, a, b) | u.data의 데이터를 정렬하여 훈련데이터와 테스트데이터로 나누어 둔  데이터 |

이 중에서 우리는 rating data가 포함된 u.base와 u.test데이터만 사용할 것이다. 유저의 많은 특성을 가장 포괄적으로 아우를 수 있는 게 바로 rating data이고 이 값을 통해 유저에게 가장 직관적으로 영화를 추천해 줄 수 있기 때문이다. 물론 사용자에 관한 데이터를 이용하여 K-mean clustering 등으로 더욱 정밀한 유사도 분석이 가능할 수도 있겠지만, 본 프로젝트에서는 평점만 이용하는 전통적인 협업 필터링 기법을 소개하고자 한다.



u.base와 u.test의 각 열의 대한 설명은 다음 표와 같다.

| 열        | 내용                                   | 사용  여부 |
| --------- | -------------------------------------- | ---------- |
| userId    | 사용자의  ID                           | o          |
| movieId   | 영화의  ID                             | o          |
| rating    | 사용자가  영화에 남긴 평점, 1~5의 정수 | o          |
| timestamp | 평점을  남긴 시간                      | x          |






# 3. Methodology

​	영화 추천 시스템에서 사용하는 방식은 크게 Collaborative Filtering(협업 필터링)과 Contents based recommendation(내용 기반 추천)으로 구분된다. 두 방식 중 Collaborative Filtering 방식은 추천의 정확도가 높다는 장점 때문에 Contents based recommendation 방식 보다 더욱 널리 사용되고 있다.

​	추천 시스템에서의 Collaborative Filtering 방식은 모든 사용자의 데이터를 균일하게 사용하는 것이 아니라 평점 행렬이 가진 특정한 패턴을 찾아서 이를 평점 예측에 사용하는 방법이다. 많은 유저들로부터 모은 취향 정보들을 기반으로 하여 스스로 예측하는 기술을 말한다. 사용자들의 평가 정보를 사용하여 데이터 베이스를 구축하고 목표 사용자와 유사한 선호도를 가진 사용자들을 데이터베이스로부터 찾아내어 이들의 선호에 기반해 새로운 평가를 추측하고 이를 목표 사용자에게 추천하는 방식이다. 예를 들어, 어떤 특정한 인물 A가 한가지 이슈에 관해서 인물 B와 같은 의견을 갖는다면 다른 이슈에 대해서도 비슷한 의견을 가질 확률이 높을 것이라는 사실에 기반한다.

​	Collaborative Filtering의 추천 시스템은 유사도를 기반으로 동작한다. 사용자-사용자 간의 유사도를 기준으로 하는 경우는 사용자 기반(User-Based), 아이템-아이템 간의 유사도를 기준으로 하는 경우는 [아이템 기반 (Item-Based)](https://en.wikipedia.org/wiki/Item-item_collaborative_filtering)이다. 아마존과 넷플릭스를 비롯한 서비스에서는 대부분 아이템 기반을 활용한다고 알려져 있다. 또 적절한 UX를 위해서라면, 유저가 아이템을 평가하는 순간, 다른 아이템을 추천할 수 있어야 하는데, 매 평가시마다 유사도 정보를 업데이트하는 것은 현실적으로 어려울 것이다. 아이템 기반에서는 일정 기간마다 유사도를 업데이트 하는 것으로도 충분히 위와 같은 적절한 UX를 제공할 수 있다. 또 대부분의 경우는 사용자에 비해 아이템 수가 적기 때문에, 아이템 간의 관계 데이터가 발생할 확률이 높다. 따라서 데이터가 누적될수록 추천의 정확도가 높아질 가능성이 더 높다. 



------



이 글에서는 사용자 기반의 유사도를 이용한 Collaborative Filtering을 예로 들어 설명할 것이다.

우선 간단한 예시 상황을 가정하도록 한다. 다음은 어떤 사용자들이 영화의 평점을 (0~5) 사이의 값으로 매긴 결과이다. 빈칸은 평가를 하지 않은 항목이다.

|      | 공조 | 더 킹 | 라라랜드 | 컨택트 | 너의 이름은 |
| ---- | ---- | ----- | -------- | ------ | ----------- |
| 재석 | 5    | 4     | 4        | 3      |             |
| 명수 | 1    | 0     | 1        |        | 4           |
| 하하 | 4    | 4     |          | 5      | 3           |
| 준하 |      | 2     | 1        | 4      | 3           |
| 세형 | 4    |       | 4        | 4      | 2           |
| 광희 | 4    | 2     | 3        |        | 1           |

사용자 기반의 협업 필터링에서의 유사도는 두 사용자가 얼마나 유사한 항목(아이템)을 선호했는지를 기준으로 한다. 한 사용자가 평가한 영화들의 점수들을 벡터로 나타낼 수 있다. 위의 예시에서 재석의 평가 점수는 <5, 4, 4, 3, ->로 나타낼 수 있다. 이 때 두 사용자 간의 유사도는 두 벡터 간의 유사도로 정의할 수 있다. 두 벡터 간의 유사도를 구하기 위해서 다양한 방법이 사용될 수 있는데 대개는 [코사인 유사도](https://en.wikipedia.org/wiki/Cosine_similarity), [피어슨 유사도](https://en.wikipedia.org/wiki/Correlation_and_dependence)가 이용된다.

코사인 유사도는 두 벡터 간의 코사인 각도를 이용하여 구할 수 있는 두 벡터의 유사도를 의미한다. 두 벡터의 방향이 완전히 동일한 경우에는 1의 값을 가지며, 90°의 각을 이루면 0, 180°로 반대의 방향을 가지면 -1의 값을 갖게 된다.  즉, 결국 코사인 유사도는 -1 이상 1 이하의 값을 가지며 값이 1에 가까울수록 유사도가 높다고 판단할 수 있다. 이를 직관적으로 이해하면 두 벡터가 가리키는 방향이 얼마나 유사한가를 의미한다. 



![](https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/co0.png)





두 벡터 A, B에 대해서 코사인 유사도는 식으로 표현하면 다음과 같다. 

![sim2]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/sim2.png )

코사인 유사도를 이용해 명수와 준하의 유사도를 구해보자. 유사도를 구할 때에는 **두 사용자가 공통으로 평가한 항목**에 대해서만 계산한다. A=(0, 1, 4), B=(2, 1, 3)이고, 코사인 유사도는 다음과 같이 정의된다.

![sim2]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/sim.png )

이런 식으로 모든 사용자에 대해서 유사도를 구하면, 사용자 수 * 사용자 수 (6*6) 크기의 유사도 행렬을 만들 수 있다.

| 유사도 | 재석 | 명수 | 하하 | 준하 | 세형 | 광희 |
| ------ | ---- | ---- | ---- | ---- | ---- | ---- |
| 재석   | 1.00 | 0.84 | 0.96 | 0.82 | 0.98 | 0.98 |
| 명수   | 0.84 | 1.00 | 0.61 | 0.84 | 0.63 | 0.47 |
| 하하   | 0.96 | 0.61 | 1.00 | 0.97 | 0.99 | 0.92 |
| 준하   | 0.82 | 0.84 | 0.97 | 1.00 | 0.85 | 0.71 |
| 세형   | 0.98 | 0.63 | 0.99 | 0.85 | 1.00 | 0.98 |
| 광희   | 0.98 | 0.47 | 0.92 | 0.71 | 0.98 | 1.00 |

서로에 대한 유사도를 안다면 세형이 아직 보지 않은 영화인 "더 킹"에 대한 평가 점수를 예측할 수 있다. 전체를 대상으로 유사도 기반의 가중 평균 값을 이용해 세형의 더 킹에 대한 평점을 예측해보면 다음과 같다. 

![sim2]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/pre.png )



------

실제 알고리즘에선 보통 한 유저와 가장 유사한 K명의 점수를 이용하는 K-Nearest Neighbor 기법을 사용한다. 

최근접 이웃의 수를 N이라고 하고 해당 유저와 i번째 최근접 이웃과의 유사도를 sim_i, 그 이웃이 준 평점을 R_i라고 할 때, 한 유저의 해당 영화에 대한 예상 평점 R_predict은 다음의 식으로 표현할 수 있다. **평점들을 단순히 가중 평균한 것이다.**

![]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/KNN.png )



------

앞서 설명한 코사인 유사도 공식 대신 사용자의 평균 평점값을 이용해 더 최적화된 공식을 사용할 수도 있다.

두 유저의 영화 평점 벡터(5, 5, 5, 5, 5, 5), (1, 1, 1, 1, 1, 1) 의 유사도를 계산해보면 1이다. 영화에 대한 유저의 평가는 극명하게 갈리는데, 이 둘의 유사도가 1이라고 한다. 이렇게 코사인 유사도에서는 유저마다의 개인적인 평가 성향을 반영하지 못한다는 단점이 있다. 이를 보완하기 위해 기존 코사인 유사도에 약간의 보정 과정을 거친, **피어슨 유사도(Pearson Similarity)**를 사용할 수도 있다. 피어슨 유사도는 두 벡터의 상관계수(Pearson correlation coefficient)를 의미한다. 유사도가 가장 높을 경우 값이 1, 가장 낮을 경우 -1의 값을 가진다. 특정 유저의 점수기준이 극단적으로 너무 낮거나 높을 경우 유사도에 영향을 크게 주기 때문에, 이를 막기 위해 상관계수를 사용하는 방법이다. 아래의 식은 사용자 u와 사용자 v간의 피어슨 유사도이다.

![sim1]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/pear.png )

피어슨 유사도에서는 사용자의 평점값에서 사용자의 평점의 평균값을 빼줌으로써, 위의 문제를 어느 정도 해결한다.



------

피어슨 유사도의 방식과 비슷하게 앞서 설명한 KNN방식으로 평점을 가중 평균하는 공식도 보정할 수 있다. 유저의 평점을 유저 평점의 평균값으로 보정하여 다음과 같은 식을 사용한다. **평점들을 평균값 기준으로 가중 평균한 KNN with Means 기법이다.**

​    ![]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/KNNwithmean.png )






# 4. Evaluation & Analysis

사용자 기반 협업 필터링 알고리즘을 코드로 작성해 실험해보았다.

원래 데이터셋을 8대 2로 나눈 u1.base, u1.test를 각각 training data, test data로 이용한다.

| 열        | 내용                                   | 사용  여부 |
| --------- | -------------------------------------- | ---------- |
| userId    | 사용자의  ID                           | o          |
| movieId   | 영화의  ID                             | o          |
| rating    | 사용자가  영화에 남긴 평점, 1~5의 정수 | o          |
| timestamp | 평점을  남긴 시간                      | x          |

------



```python
import pandas as pd
import numpy as np

N_user = 943
N_movie = 1682

data = pd.read_csv("u1.base", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
del data['timestamp']
data = np.array(data)

# ratings = { user_id : { movie_id: rating } }
ratings = {i:{} for i in range(1,N_user+1)}
for a,b,c in data:
    ratings[a][b] = c

# neighbors = { user_id : [sorted (user_id, similarity)] }
neighbors = {}
# means = { user_id : mean of user's ratings}
means = {}
for i in ratings:
    rat = 0
    count = 0
    for j in ratings[i]:
        count += 1
        rat += ratings[i][j]
    means[i] = rat / count    
```

총 사용자 수는 943명이고 총 영화 수는 1682개이다. 이 경우엔 사용자 수가 영화 수보다 적으니 사용자 기반 추천이 성능이 더 높을 것이다.

"u1.base" 파일에서 userId, movieId, rating 정보를 가져온다.

"ratings" 자료구조에 각 user 의 user가 본 모든 영화에 대한 rating 을 저장한다. 

"neighbors"는 각 user에 대해서 그 유저를 제외한 모든 유저와의 유사도를 저장한다. 이때, 유사도가 높은 유저 순서로 정렬한다. (KNN)

"means" 자료구조엔 각 유저에 대한 해당 유저가 매긴 평점의 평균값을 저장한다. 나중에 유저간 유사도를 계산하거나 평점을 예측할 때 쓰인다.

------

**유저간 유사도 구하는 함수**

```python
def sim_user(user1,user2):
       
    rating1 = []
    rating2 = [] #두 유저의 평점을 저장할 리스트
    
    #두 사람이 모두 본 영화만 리스트로 저장
    for movie_id in ratings[user1]:
        if movie_id in ratings[user2]:
            rating1.append(ratings[user1][movie_id])
            rating2.append(ratings[user2][movie_id])
      
    if len(rating1) == 0:
        return 0.0
    
    for x in range(len(rating1)):
        rating1[x] = rating1[x] - means[user1]
        rating2[x] = rating2[x] - means[user2] #각 평점에서 평점의 평균을 빼준다.

    vec = ((np.linalg.norm(rating1))*(np.linalg.norm(rating2))) #두 평점의 크기를 미리 곱한다.

    if vec != 0.0: #분모가 0이 아니라면
        sim = np.dot(rating1,rating2)/(vec)#코사인 유사도 계산
    
        return round(sim,4)#소수점 아래 4자리까지 계산후 리턴

    else:
        return 0.0#분모가 0이면 계산할 수 없다. 0.0리턴
```

두 유저간 유사도를 계산하는 함수이다. 두 유저가 모두 본 영화의 평점만 이용한다. 두 사용자의 영화 평점을 벡터로 하여 유사도를 구한다. 위 코드에선 보정된 코사인 유사도인 피어슨 유사도 (Pearson correlation  coefficient )공식을 사용한다.

![sim1]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/pear.png )



```python
print(sim_user(1,2))
print(sim_user(1,3))
print(sim_user(1,4))
```

```python
0.5218
0.5251
1.0
```

차례대로 유저1과 2의 유사도, 유저 1과 3의 유사도, 유저 1과 4의 유사도를 출력 해보았다. Pearson correlation 유사도는 -1부터 1까지 값이 가능하다.



------

**각 유저의 다른 모든 유저에 대한 similarity 계산해서 저장하는 함수**

```python
def calculate_similarity():
    
    # 1 ~ 943(N_user)
    for i in range(1, N_user+1):
        nei = []
        for j in range(1, N_user+1):
            if i != j:
                nei.append((j, sim_user(i, j)))
                   
        # 네이버 유저의 similarity 기준 내림차순으로 정렬            
        nei.sort(key=lambda x: x[1], reverse=True)
        neighbors[i] = nei
        # neighbors = { user_id : [sorted (user_id, similarity)] }
```

이 함수를 실행하면 모든 유저 조합간 유사도를 계산한다. 즉, 각 943명의 유저에 대해 해당 유저를 제외한 다른 모든 유저(942명)와의 유사도를 계산하고, 유사도가 큰 것이 앞에 나오게 정렬해 `neighbors` 자료구조에 저장한다.  유사도 대로 정렬하는 이유는 rating 예측시 KNN 기법을 사용하는데, 쉽게 제일 가까운(유사도가 제일 큰) 이웃 유저를 접근할 수 있기 때문이다.

------

유저2와 다른 모든 유저간 유사도가 정렬된 리스트가 저장되어 있는 것을 확인해본다.

```
calculate_similarity()   # 모든 유저간 유사도 계산해서 저장
print(len(neighbors[2])) # 유저2의 다른 942명 유저와의 유사도
print(neighbors[2])
```

```
942
[(5, 1.0), (22, 1.0), (25, 1.0), (67, 1.0), (76, 1.0), (96, 1.0), (98, 1.0), (135, 1.0), (142, 1.0), (154, 1.0), (156, 1.0), (208, 1.0), (217, 1.0), (260, 1.0), (270, 1.0), (290, 1.0), (310, 1.0), (340, 1.0), (359, 1.0), (366, 1.0), (367, 1.0), (368, 1.0), (471, 1.0), (519, 1.0), (522, 1.0), (607, 1.0), (686, 1.0), (911, 1.0), (912, 1.0), (167, 0.9997), (744, 0.999), (80, 0.9951), (267, 0.9939), (600, 0.9928), (645, 0.9895), (23, 0.9666)
.. 
 (28, -0.9809), (97, -0.9839), (248, -0.9969), (8, -1.0), (20, -1.0), (31, -1.0), (41, -1.0), (124, -1.0), (127, -1.0), (148, -1.0), (180, -1.0), (182, -1.0), (187, -1.0), (211, -1.0), (224, -1.0), (225, -1.0), (282, -1.0), (292, -1.0), (317, -1.0), (565, -1.0), (712, -1.0), (849, -1.0), (855, -1.0), (914, -1.0), (925, -1.0)]
```



------

**유저(user_id)의 영화(movie_id) 평점 예측  **

```python
def predict_rating(user_id, movie_id):
    rating = 0
    K = 0
    j = 0
    for i in range(N_user - 1):
        # valid neighbor 40개까지
        if j > 40:
            break
        # 해당 영화 평점을 실제로 매긴 neighbor 유저만 취급
        if movie_id in ratings[neighbors[user_id][i][0]]:
            j += 1
            nei_id = neighbors[user_id][i][0]
            nei_sim = neighbors[user_id][i][1]
            
            rating += ratings[nei_id][movie_id] * nei_sim
            K += nei_sim
    
    if K != 0:
        rating = (rating / K) 
    else:
        rating = 2.3
        
    return rating
```

이 함수는 한 유저의 한 영화에 대한 평점을 예측해 리턴하는 함수이다. 

"K-Nearest Neighbors"유저의 평점을 유사도로 가중 평균내는 공식을 사용하여 예측 평점을 계산한다. 

![]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/KNN.png )

유저와 제일 가까운 K개 (위에선 40개) 이웃 유저의 평점을 활용한다. 이 때, 해당 영화 평점 데이터가 없는 유저가 있을 수 있기 때문에 예외처리를 잘 해야한다. 해당 영화의 평점을 실제로 매긴 40개 이웃 유저의 평점과 유저와 이웃 유저 간 유사도를 이용해 공식처럼 계산하면 예측값을 얻을 수 있다.

------

```python
print(predict_rating(1,1))
print(predict_rating(1,2))
```

```
4.4178939023402055
3.3305233814943893
```

이렇게 유저1의 영화1에 대한 예측평점과 유저1의 영화2에 대한 예측평점을 출력해보면 1과 5 사이로 예측한다는 것을 알 수 있다.

------

```python
def prediction():
    test = pd.read_csv("u1.test", sep="\t", names=['userId', 'movieId', 'rating', 'timestamp'])
    del test['timestamp']
    test = np.array(test)
    
    sqaure = 0
    #  RMSE (Root Mean Square Error) 
    for user_id, movie_id, rating in test:
        # sum of (아이템의 예측 레이팅 - 아이템의 원래 레이팅) ** 2
        sqaure += (predict_rating(user_id, movie_id) - rating) ** 2
    return np.sqrt(sqaure / len(test))
```

이 함수는 테스트 파일의 모든 데이터 레코드 (유저id, 영화id, 평점)에 대해서 해당 유저의 해당 영화 예측 평점을 구해 실제 평점과의 차로 오차를 계산한다. 오차는 RMSE (Root Mean Square Error) 를 사용한다. R_real 은 실제 rating 값이고 R_predict은 예측 rating 값이며 N은 rating의 수 (혹은 레코드 수)이다.

![]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/rmse.png )

리턴값인 RMSE는 말그대로 오류이므로 더 작을수록 모델의 예측 성능이 더 뛰어난 것이다.

```
rmse = prediction()
print(rmse)
```

```
1.0463047473120506
```

전체 테스트 데이터셋의 RMSE는 0.986172895393817 이 나왔다.

------



```python
def predict_rating(user_id, movie_id):
    rating = 0
    K = 0
    j = 0
    for i in range(N_user - 1):
        # valid neighbor 40개까지
        if j > 40:
            break
        # 해당 영화 평점을 실제로 매긴 neighbor 유저만 취급
        if movie_id in ratings[neighbors[user_id][i][0]]:
            j += 1
            nei_id = neighbors[user_id][i][0]
            nei_sim = neighbors[user_id][i][1]
                                                 ### mean으로 보정
            rating += (ratings[nei_id][movie_id] - means[nei_id]) * nei_sim
            K += nei_sim
    
    if K != 0:
        rating = means[user_id] + (rating / K)  ###
    else:
        rating = 2.3
        
    return rating
```

만약 `predict_rating` 함수에서 "KNN 가중 평균" 방식을 사용하지 않고 아래 공식처럼  "KNN with means"방식으로 평점들을 평균값 기준으로 가중 평균 했을시엔 

![]( https://raw.githubusercontent.com/skifree64/skifree64.github.io/master/_posts/KNNwithmean.png )

```
rmse = prediction()
print(rmse)
```

```
0.986172895393817
```

다음과 같은 결과가 나온다.

```
rmse(knn) => 1.0463047473120506
rmse(knnWithMeans) => 0.986172895393817
```

평점 예측하는 방식을 mean 값으로 보정하니 RMSE가 줄어들고 더 최적화된 예측을 한다는 것을 알 수 있다.

같은 이유로 유사도를 구할 때 그냥 코사인 유사도보다 피어슨 유사도를 사용하는게 성능이 더 좋을 것이다. 

 

# 5. Related Work

[1] https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

[2] http://www.grouplens.org/node/12

[3] 이재식, 박석두 (2007). 장르별 협업필터링을 이용한 영화추천시스템의 성능 향상. 지능정보연구, 13(4), 65-7

[4] 김부성, 김희라, 이재동, 이지형 (2013). 사용자 개인정보를 이용한 협업 필터링 기반 영화 추천 시 스템. 한국지능시스템학회 학술발표 논문집, 23(2), 63-64

[5] 윤소영, 윤성대 (2011). 사용자 정보 가중치를 이용한 추천 기법. 한국정보통신학회논문지, 15(4), 877-88






# 6. Conclusion: Discussion

소비자가 어떤 결정을 할 때 수많은 정보를 다 고려하는 것은 시간과 비용이 많이 소요된다. 소비자에게 가장 알맞은 정보를 제공하면 시간과 비용을 절약할 수 있을 뿐 아니라 더욱 합리적인 선택에 도움이 될 것이다. 경쟁이 치열해지는 기업환경에서 제품이나 서비스 제공자들이 고객 개개인에게 맞춤 아이템이나 정보를 제공하는 것은 매출신장에 있어 중요한 요인 중의 하나이다. 이런 개인화 서비스를 가능하게 하는 방법의 하나가 추천시스템이다 . 추천시스템에서 사용되는 여러 기법들 중에서 전자상거래에서 성공으로 적용된 대표인적인 기법은 협업 필터링이다. 협업 필터링은 명시적인 속성만으로 규정짓기 힘든 동영상이나 음악 같은 아이템 들에도 효과인 성능을 발휘한다는 장점이 있다. 협업 필터링을 이용한 추천 시스템으로 사람들의 신뢰를 얻는다면, 더 많은 사용자의 정보를 얻을 수 있고, 더욱 정교한 추천 시스템을 만들 수 있을 것이다.