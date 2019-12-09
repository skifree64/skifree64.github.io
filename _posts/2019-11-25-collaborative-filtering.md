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

​	이번 프로젝트에서는 협업 필터링에 대해서 소개하고 직접 협업 필터링을 통해 영화를 추천하는 프로그램을 만들 예정이다. 최종적으로, 학습데이터의 양이 많을 때와 적을 때에 따라, 유사도를 구하는 대표적인 두가지 방법 중 어느 것을 사용했는 지에 따라 프로그램의 예측 성능이 어떻게 달라지는지 알아보려고 한다.




# 2. Datasets

 1. 데이터 셋의 내용

    우리가 사용할 데이터 셋은 ‘MovieLens Data’이다. Group Lens(http://www.grouplens.org/node/12)

    라는 미네소타 대학의 컴퓨터과의 연구실에서 수집한 추천 알고리즘을 위한 영화 데이터이다. 여기에는 데이터의 양에 따라 데이터셋이 다양하게 있는데, 그 중에 100K DataSet을 사용할 것이다(http://www.grouplens.org/system/files/ml-100k.zip). 이 데이터 셋은 총20가지의 데이터로 이루어져 있다. 각 데이터의 내용은 다음의 표와 같다.

    | 파일명                     | 데이터 내용                                                  |
    | -------------------------- | ------------------------------------------------------------ |
    | u.data                     | 943명의 사용자가 1682개의  영화에 남긴 100000개의 평점 데이터. 무작위 배열 |
    | u.info                     | u.data의 사용자의 수, 영화의  수, 평점의 수                  |
    | u.item                     | 이 데이터에서 사용된 영화의 정보                             |
    | u.genre                    | 장르와 각 장르의 코드                                        |
    | u.user                     | 유저의 ID, 나이, 성별, 직업, 주소                            |
    | u.occupation               | 직업의 종류                                                  |
    | u.base, u.test (1~5, a, b) | u.data의 데이터를 정렬하여 훈련데이터와 테스트데이터로 나누어 둔  데이터 |

    이 중에서 우리는 rating data가 포함된 u.base와 u.test데이터만 사용할 것이다. 유저의 많은 특성을 가장 포괄적으로 아우를 수 있는 게 바로 rating data이고 이 값을 통해 유저에게 가장 직관적으로 영화를 추천해 줄 수 있기 때문이다. 물론 사용자에 관한 데이터를 이용하여 K-mean clustering 등으로 더욱 정밀한 유사도 분석이 가능할 수도 있겠지만, 본 프로젝트에서는 평점을 이용하는 전통적인 협업 필터링 기법을 소개하고자 한다.

    그리고 movieId에 대응하는 영화가 무엇인지 확인하기 위해 u.item도 참고한다.

    u.base와 u.test의 각 열의 대한 설명은 다음 표와 같다.

    | 열        | 내용                                   | 사용  여부 |
    | --------- | -------------------------------------- | ---------- |
    | userId    | 사용자의  ID                           | o          |
    | movieId   | 영화의  ID                             | o          |
    | rating    | 사용자가  영화에 남긴 평점, 1~5의 정수 | o          |
    | timestamp | 평점을  남긴 시간                      | x          |

	2.  데이터 셋 전처리

    이 데이터의 모든 파일은 tab("\t")을 기준으로 열이 분리되어 있다. 이런 형식의 데이터를 pandas로 읽어 들이기 위해선 다음과 같은 코드를 입력하면 된다.  

    `data = pd.read_csv([file_name], sep="\t", names=[column_list])`

    [file_name]에 읽어드릴 파일의 이름, [column_list]에 열 이름을 리스트로 전달하면 된다.

    하지만 열람하기 쉽도록 csv파일로 변환하고자 한다. 다음의 코드를 실행시키면 u.base와 u.test를 csv파일로 변환시킬 수 있다.

    (Ratingdata.py)

    읽어들이는 파일의 이름과, 내보내는 파일의 이름을 수정하면 7쌍의 u.base, u.test를 csv파일로 변환시킬 수 있다.

    u.item을 이용해서 영화 데이터도 csv파일로 전환한다.

    (Moviedata.py)






# 3. Methodology

​	영화 추천 시스템에서 사용하는 방식은 크게 Collaborative Filtering(협업 필터링)과 Contents based recommendation(내용 기반 추천)으로 구분된다. 두 방식 중 Collaborative Filtering 방식은 추천의 정확도가 높다는 장점 때문에 Contents based recommendation 방식 보다 더욱 널리 사용되고 있다.

​	추천 시스템에서의 Collaborative Filtering 방식은 모든 사용자의 데이터를 균일하게 사용하는 것이 아니라 평점 행렬이 가진 특정한 패턴을 찾아서 이를 평점 예측에 사용하는 방법이다. 많은 유저들로부터 모은 취향 정보들을 기반으로 하여 스스로 예측하는 기술을 말한다. 사용자들의 평가 정보를 사용하여 데이터 베이스를 구축하고 목표 사용자와 유사한 선호도를 가진 사용자들을 데이터베이스로부터 찾아내어 이들의 선호에 기반해 새로운 평가를 추측하고 이를 목표 사용자에게 추천하는 방식이다. 예를 들어, 어떤 특정한 인물 A가 한가지 이슈에 관해서 인물 B와 같은 의견을 갖는다면 다른 이슈에 대해서도 비슷한 의견을 가질 확률이 높을 것이라는 사실에 기반한다.

​	Collaborative Filtering의 추천 시스템은 유사도를 기반으로 동작한다. 사용자-사용자 간의 유사도를 기준으로 하는 경우는 사용자 기반(User-Based), 아이템-아이템 간의 유사도를 기준으로 하는 경우는 [아이템 기반 (Item-Based)](https://en.wikipedia.org/wiki/Item-item_collaborative_filtering)이다. 아마존과 넷플릭스를 비롯한 서비스에서는 대부분 아이템 기반을 활용한다고 알려져 있다. 또 적절한 UX를 위해서라면, 유저가 아이템을 평가하는 순간, 다른 아이템을 추천할 수 있어야 하는데, 매 평가시마다 유사도 정보를 업데이트하는 것은 현실적으로 어려울 것이다. 아이템 기반에서는 일정 기간마다 유사도를 업데이트 하는 것으로도 충분히 위와 같은 적절한 UX를 제공할 수 있다. 또 대부분의 경우는 사용자에 비해 아이템 수가 적기 때문에, 아이템 간의 관계 데이터가 발생할 확률이 높다. 따라서 데이터가 누적될수록 추천의 정확도가 높아질 가능성이 더 높다. 

먼저 아이템 기반의 협업 필터링을 설명하기 앞서 간단한 예시 상황을 가정하도록 한다. 사용자 기반의 협업 필터링의 예시 상황은 생략한다. 사용자 기반은 아이템 기반에서 주어와 목적어를 서로 바꾼 것이다. 다음은 어떤 사용자들이 영화의 평점을 (0~5) 사이의 값으로 매긴 결과이다. 빈칸은 평가를 하지 않은 항목이다. 

사용자 기반에서는 사용자에 대한 유사도를 계산한다면, 아이템 기반에서는 아이템들에 대한 유사도를 계산한다.

|      | 공조 | 더 킹 | 라라랜드 | 컨택트 | 너의 이름은 |
| ---- | ---- | ----- | -------- | ------ | ----------- |
| 재석 | 5    | 4     | 4        | 3      |             |
| 명수 | 1    | 0     | 1        |        | 4           |
| 하하 | 4    | 4     |          | 5      | 3           |
| 준하 |      | 2     | 1        | 4      | 3           |
| 세형 | 4    |       | 4        | 4      | 2           |
| 광희 | 4    | 2     | 3        |        | 1           |

두 아이템에 대한 사용자들의 평가 점수를 벡터로 나타내보자. 공조와 라라랜드의 유사도를 구하려고 한다면, 이 둘을 모두 평가한 사용자는, 재석, 명수, 세형, 광희다. 각각 (5, 1, 4, 4)와 (4, 1, 4, 3)이다.

![sim0](C:\Users\Hyunsik Yoo\Github\skifree64.github.io\_posts\sim0.png)

![sim0]( https://github.com/skifree64/skifree64.github.io/blob/master/_posts/sim0.png )

공조와 라라랜드의 유사도는 0.99로 상당히 높은 유사도를 보인다. 이는 즉 공조를 좋아하는 사람은 라라랜드를 좋아할 확률이 높다는 말로 풀이 될수 있고, 그 반대로도 해석할 수 있다.

아이템 기반에서도 앞처럼 아이템 수 * 아이템 수 (5*5) 크기의 유사도 행렬을 만들 수 있다. 이를 기반으로 역시 예측 점수를 구할 수 있다.

두 벡터 간의 유사도를 구하기 위해서 다양한 방법이 사용될 수 있는데 대개는 코사인 유사도, 피어슨 유사도가 이용된다.

코사인 유사도는 다음과 같이 정의된다.

![sim2](C:\Users\Hyunsik Yoo\Github\skifree64.github.io\_posts\sim2.png)



다음의 두 아이템의 유사도를 구해보자, (5, 5, 5, 5, 5, 5), (1, 1, 1, 1, 1, 1). 이 때, 이 둘 간의 유사도는 1이다. 두 아이템에 대한 사용자들의 평가는 극명하게 갈리는데, 이 둘의 유사도가 1이라고 한다. 이렇게 코사인 유사도에서는 유저마다의 개인적인 평가 성향을 반영하지 못한다는 단점이 있다. 이를 보완하기 위해, 앞서 말한 피어슨 유사도, 혹은 약간의 보정 과정을 거친 코사인 유사도를 사용할 수도 있다.

![sim1](C:\Users\Hyunsik Yoo\Github\skifree64.github.io\_posts\sim1.png)



보정된 코사인 유사도에서는 사용자의 평균 평가 값을 빼줌으로써, 위의 문제를 어느 정도 해결할 수 있다.

​	사실 이 외에도 몇 가지 문제점이 더 있다. 예를 들어, 어떤 두 아이템에 대해서 유사도를 구하려 하는데, 이 두 영화를 모두 평가한 사람이 한 명인 경우에 유사도가 역시 1이 나오게 된다. 이렇게 적은 데이터에 기반한 정보는 추천의 정확도를 떨어뜨릴 가능성이 높다. 이런 경우를 방지하기 위해 할 수 있는 조치는, 최소 평가 인원의 수를 정하는 것이다. 두 아이템을 모두 평가한 사람의 수가 5명 이상일 때부터 유사도를 계산하기로 하고, 5명 미만의 공통 평가 인원수를 갖는 두 아이템에 대한 유사도를 0으로 처리하는 것이다.

 	아마존과 넷플릭스를 비롯한 서비스에서는 대부분 아이템 기반을 활용한다고 알려져 있다. 또 적절한 UX를 위해서라면, 유저가 아이템을 평가하는 순간, 다른 아이템을 추천할 수 있어야 하는데, 매 평가시마다 유사도 정보를 업데이트하는 것은 현실적으로 어려울 것이다. 아이템 기반에서는 일정 기간마다 유사도를 업데이트 하는 것으로도 충분히 위와 같은 적절한 UX를 제공할 수 있다. 또 대부분의 경우는 사용자에 비해 아이템 수가 적기 때문에, 아이템 간의 관계 데이터가 발생할 확률이 높다. 따라서 데이터가 누적될수록 추천의 정확도가 높아질 가능성이 더 높다. 

​	

​	유사도를 다 구했다면, 우리의 다음 관심사는 평점을 예측하는 방법이다. 평점 예측의 가장 대표적인 방법은 평점을 유사도로 가중 평균하는 방법이다. 아이템 기반에서 User1이 movie1에 줄 평점을 예측하는 상황을 가정하자. 우선, User1이 평점을 준 모든 영화와 movie1의 유사도를 계산한다. 그리고 User1이 각 영화에 준 평점과 유사도를 곱하여 평균을 내리는 것이다. 사용자 기반에서는 영화와 사용자를 바꾸어 생각하면 되므로 설명을 생략하겠다. 이 가정에서는 모든 영화와 movie1의 유사도를 구했지만 모든 영화와의 유사도를 이용하면 오버 피팅의 가능성이 커진다. 따라서 보통 movie1과의 유사도가 큰 영화만을 이용한다. 유사도가 큰 데이터를 고르는 방법에는 대표적으로 두 가지가 있다. 첫 번째는 미리 설정한 수치만큼의 유사도를 넘어야만 예측에 사용하는 방법이다. 그리고 두 번째는 K-nn(K-nearest neighbors)알고리즘을 사용하여 최근접 이웃을 구성하는 방법이다. 이 과정에서 구한 최근접 이웃의 수를 N이라고 하고 i번째 최근접 이웃과의 유사도를 n_i, 그 이웃이 준 평점을 R_i라고 할 때, 예상 평점 R_predict은 다음의 식으로 표현할 수 있다.
$$
R_{predict}=(∑_{i=1}^Nn_i R_i)/(∑_{i=1}^Nn_i )
$$


일반적으로는, 피어슨 유사도와 같이 mean으로 보정하여 다음과 같은 식을 사용한다.

​    
$$
R_{predict}=(R_{real}) ̅+(∑_{i=1}^Nn_i (R_i-R_{mean} )) / (∑_{i=1}^Nn_i )
$$


# 4. Evaluation & Analysis



사용자 기반 협업 필터링 알고리즘을 코드로 작성해보았다.

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

두 유저간 유사도를 계산하는 함수이다. 두 유저가 모두 본 영화의 평점만 이용한다. 두 사용자의 영화 평점을 벡터로 하여 유사도를 구한다. 위 코드에선 보정된 코사인 유사도인 "Pearson correlation"공식을 사용한다.

![sim1](C:\Users\Hyunsik Yoo\Github\skifree64.github.io\_posts\sim1.png)



```python
# 각 유저의 다른 모든 유저에 대한 similarity 계산해서 저장
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

이 함수를 실행하면 모든 유저 조합간 유사도를 계산한다. 즉, 각 943명의 유저에 대해 해당 유저를 제외한 다른 모든 유저(942명)와의 유사도를 계산하고, 유사도가 큰 것이 앞에 나오게 정렬해 "neighbors" 자료구조에 저장한다.  유사도 대로 정렬하는 이유는 rating 예측시 KNN 기법을 사용하는데, 쉽게 제일 가까운(유사도가 제일 큰) 이웃 유저를 접근할 수 있기 때문이다.



```python
# 유저(user_id)의 영화(movie_id) 평점 예측        
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
            
            rating += (ratings[nei_id][movie_id] - means[nei_id]) * nei_sim
            K += nei_sim
    
    if K != 0:
        rating = rating / K + means[user_id]
    else:
        rating = 2.3
        
    if rating < 1:
        rating = 1
    elif rating > 5:
        rating = 5
        
    return rating

```

이 함수는 한 유저의 한 영화에 대한 평점을 예측해 리턴하는 함수이다. 





# 5. Related Work

[1] https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

[2] http://www.grouplens.org/node/12

[3] 이재식, 박석두 (2007). 장르별 협업필터링을 이용한 영화추천시스템의 성능 향상. 지능정보연구, 13(4), 65-7

[4] 김부성, 김희라, 이재동, 이지형 (2013). 사용자 개인정보를 이용한 협업 필터링 기반 영화 추천 시 스템. 한국지능시스템학회 학술발표 논문집, 23(2), 63-64

[5] 윤소영, 윤성대 (2011). 사용자 정보 가중치를 이용한 추천 기법. 한국정보통신학회논문지, 15(4), 877-88






# 6. Conclusion: Discussion
