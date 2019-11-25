---
layout: post
title:  "Collaborative Filtering"
author: 유현식, 임정빈, 정연경, 이준희
date:   2019-11-25
categories: Machine Learning
tags: recommendation_system
---


[TOC]



# 협업 필터링 추천 시스템 (Collaborative Filtering Recommendation System)



# Members

**유현식, 컴퓨터전공**

**임정빈, 원자력공학과**

**정연경, 원자력공학과**

**이준희, 기계공학부**



# 1. Introduction

​	영상을 재생할 수 있는 기기들이 많이 보급되면서 영상 콘텐츠에 대한 수요는 지속적으로 늘어나고 있으며, 이에 따라 수많은 양의 영상 콘텐츠가 실시간으로 공급되고 있다. 하지만 이러한 과잉 공급 때문에, 사용자들은 자신에게 적합한 콘텐츠를 찾는데 어려움을 겪고 있다. 일부 사람들은 이전에 상영했던 영화 중 자신에게 적합한 영화를 찾기 위해 영화 정보 제공 TV 프로그램을 이용하거나 포털 사이트의 지식정보 커뮤니티를 이용하고 있다. 하지만 이러한 소통 방식들은 사용자의 취향이나 의중을 완전히 반영할 수 없고, 추천의 범위가 개인의 지식 내부로 한정되는 단점이 있기 때문에, 적절한 추천이 이루어지기 힘든 경우가 있다. 위와 같은 문제를 해결하기 하여 기업들은 추천시스템을 사용하여 사용자를 확보하고 있으며, 추천시스템은 사용자들에게 그들이 관심 있고 좋아할 만한 아이템을 추천해 주어 원하는 아이템을 쉽고 빠르게 찾을 수 있도록 돕는다. 이 시스템은 온라인 뉴스, 영화, 다양한 형태의 web resource들을 추천하며 많은 e-commerce 사이트에서 사용되고 있다. 추천 시스템이 이러한 서비스에 기여하는 바는 어마어마 하다. Netflix는 대여되는 영화의 66%, Google News는 38% 이상, Amazon은 판매의 35% 가 추천으로부터 발생하며 우리나라에서도 영상 스트리밍 사이트가 등장하면서 영화 추천 시스템이 활성화되고 있다.



# 2. Datasets

​	우리가 사용할 데이터 셋은 ‘MovieLens Data’이다. Group Lens라는 미네소타 대학의 컴퓨터과의 연구실에서 수집한 추천 알고리즘을 위한 영화 데이터이다. 이 데이터 셋은 총 3가지 파일인 ‘users.dat’, ‘movies.dat’, ‘ratings.dat’ 로 이루어져 있다. ‘users.dat’엔 약 6000명가량의 유저와 그에 따른 성, 나이, 직업 등과 같은 특징 데이터가 저장되어 있고 ‘movies.dat’(movie file)엔 약 4000개가량의 영화 정보가 저장되어 있다. ‘ratings.dat’(rating file)에는 각 유저가 자신이 본 영화에 내린 1 점에서 5점 사이의 평가 정보가 저장되어 있다. 각 유저의 rating data를 기반으로 각 유저가 선호할 만한 영화의 기준을 규정할 수 있다. 유저의 많은 특성을 가장 포괄적으로 아우를 수 있는 게 바로 rating data이고 이 값을 통해 유저에게 가장 직관적으로 영화를 추천해 줄 수 있기 때문이다.

[출처]: https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis	"ddd"
[100K Dataset]: http://www.grouplens.org/system/files/ml-100k.zip
[10M Dataset]: http://www.grouplens.org/sites/www.grouplens.org/external_files/data/ml-10m.zip
[1M Dataset]: http://www.grouplens.org/system/files/ml-1m.zip



# 3. Methodology

​	영화 추천 시스템에서 사용하는 방식은 크게 Collaborative Filtering(협업 필터링)과 Contents based recommendation(내용 기반 추천)으로 구분된다. 두 방식 중 Collaborative Filtering 방식은 추천의 정확도가 높다는 장점 때문에 Contents based recommendation 방식 보다 더욱 널리 사용되고 있다.

​	추천 시스템에서의 Collaborative Filtering 방식은 모든 사용자의 데이터를 균일하게 사용하는 것이 아니라 평점 행렬이 가진 특정한 패턴을 찾아서 이를 평점 예측에 사용하는 방법이다. 많은 유저들로부터 모은 취향 정보들을 기반으로 하여 스스로 예측하는 기술을 말한다. 사용자들의 평가 정보를 사용하여 데이터 베이스를 구축하고 목표 사용자와 유사한 선호도를 가진 사용자들을 데이터베이스로부터 찾아내어 이들의 선호에 기반해 새로운 평가를 추측하고 이를 목표 사용자에게 추천하는 방식이다. 예를 들어, 어떤 특정한 인물 A가 한가지 이슈에 관해서 인물 B와 같은 의견을 갖는다면 다른 이슈에 대해서도 비슷한 의견을 가질 확률이 높을 것이라는 사실에 기반한다.

​	Collaborative Filtering의 추천 시스템은 유사도를 기반으로 동작한다. 사용자-사용자 간의 유사도를 기준으로 하는 경우는 사용자 기반(User-Based), 아이템-아이템 간의 유사도를 기준으로 하는 경우는 [아이템 기반 (Item-Based)](https://en.wikipedia.org/wiki/Item-item_collaborative_filtering)이다. 아마존과 넷플릭스를 비롯한 서비스에서는 대부분 아이템 기반을 활용한다고 알려져 있다. 또 적절한 UX를 위해서라면, 유저가 아이템을 평가하는 순간, 다른 아이템을 추천할 수 있어야 하는데, 매 평가시마다 유사도 정보를 업데이트하는 것은 현실적으로 어려울 것이다. 아이템 기반에서는 일정 기간마다 유사도를 업데이트 하는 것으로도 충분히 위와 같은 적절한 UX를 제공할 수 있다. 또 대부분의 경우는 사용자에 비해 아이템 수가 적기 때문에, 아이템 간의 관계 데이터가 발생할 확률이 높다. 따라서 데이터가 누적될수록 추천의 정확도가 높아질 가능성이 더 높다. 



# 4. Evaluation & Analysis

*1) MovieLens Data 가져오기*

0. 환경설정

```R
library(tidyverse)
library(skimr)
library(lubridate)
library(stringr)
library(rvest)
library(XML)
library(tidytext)
library(wordcloud)
library(ggthemes)
library(extrafont)
loadfonts()
library(doParallel)

```

1. 데이터 가져오기

   1.1 가져올 데이터 설정

   ```R
   url <- "http://files.grouplens.org/datasets/movielens/"
   dataset_small <- "ml-latest-small"
   dataset_full <- "ml-latest"
   data_folder <- "data"
   archive_type <- ".zip"
   
   ```





# 5. Related Work

[1] https://www.kaggle.com/jneupane12/analysis-of-movielens-dataset-beginner-sanalysis

[2] http://www.grouplens.org/node/12

[3] 이재식, 박석두 (2007). 장르별 협업필터링을 이용한 영화추천시스템의 성능 향상. 지능정보연구, 13(4), 65-7

[4] 김부성, 김희라, 이재동, 이지형 (2013). 사용자 개인정보를 이용한 협업 필터링 기반 영화 추천 시 스템. 한국지능시스템학회 학술발표 논문집, 23(2), 63-64

[5] 윤소영, 윤성대 (2011). 사용자 정보 가중치를 이용한 추천 기법. 한국정보통신학회논문지, 15(4), 877-88





# 6. Conclusion: Discussion
