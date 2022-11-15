import operator
import numpy as np

# IDF 계산을 위해
from math import log

# 설치한 konlpy 외부라이브러리로부터 Hannanum 기능 사용하도록 설정
from konlpy.tag import Hannanum

# 문자열(문장) 수정을 위한 파이썬 기본 기능 추가
import re

# tf 함수 정의
def tf(t, d):
    return d.count(t)

# idf 함수 정의
def idf(t):
    df = 0
    for doc in docs:
        df += t in doc

    # idf에서 log 함수를 사용한 이유는 idf의 값을 가지게 하기 위해 사용함
    # idf는 기본적으로 df의 값을 1을 더함
    # 만약 문서의 수는 6개, 문서의 빈도수는 5개라면,
    # 문서의 빈도수에 1을 더하는 idf 연산을 특성상 문서의 빈도수는 5개 + 1 = 6개가 됨
    # 6 나누기 6을 하면, 분자와 분모가 같아 항상 1이 되는 문제가 발생함
    # 또한 log를 사용하지 않으면, df값에 따라 idf값은 기하급수적으로 증가함
    # 따라서 이러한 문제를 해결하기 위해 log함수를 사용함
    return log(N/(df + 1))

# tfidf 함수 정의
def tfidf(t, d):
    return tf(t, d)* idf(t)

# 유클리드 거리 알고리즘
def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))

# 형태소 분석기 사용
myHannanum = Hannanum()

# 분석할 원본 문서데이터(1개의 레코드마다 문서 1개로 설정)
org_docs = [
    "학생들은 빅데이터와 인공지능 기술을 배우고 있다.",
    "빅데이터 기술은 방대한 데이터를 처리한다. 빅데이터는 많은 데이터를 저장한다.",
    "빅데이터 기술을 많이 어렵다. 특히 하둡이 어렵다.",
    "나의 목표는 빅데이터 기술을 활용하는 빅데이터 소프트웨어 개발자이다.",
    "소프트웨어 개발은 코딩이 필수이다. 나는 소프트웨어 개발자가 되고 싶다. 소프트웨어 개발자 화이팅!",
    "인공지능 기술에서 자연어 처리는 재미있다. 자연어는 사람이 사용하는 일반적인 언어이다."

]

# 형태소 분석을 통해 변경된 문서 데이터
docs = []

# 형태소 분석기로 문서별 명사 추출하기
for org_doc in org_docs:
    replace_doc = re.sub("[!@#$%^&*()_+]", " ", org_doc)
    docs.append(" ".join(myHannanum.nouns(replace_doc)))

# 변경된 문서 출력해보기
print(docs)

# 단어별 중복 제거해서 저장하기
# set 데이터 구조 사용
vocab = list(set(w for doc in docs for w in doc.split()))
vocab.sort()

# 저장된 단어 출력
print("중복 제거된 단어 : " + str(vocab))

# 총 문서의 수
N = len(docs)

print("문서의 수 : " + str(N))

result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docs[i]

    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tfidf(t, d))

    # 문서별 빈도수를 벡터값으로 사용함
    print(result[i])

#문서1과 문서2의 유사도 분석
print("문석1과 문서2의 유사도 : "+ str(dist(np.array(result[0]), np.array(result[1]))))

#문서1과 문서3의 유사도 분석
print("문석1과 문서2의 유사도 : "+ str(dist(np.array(result[0]), np.array(result[2]))))

#문서1과 문서4의 유사도 분석
print("문석1과 문서2의 유사도 : "+ str(dist(np.array(result[0]), np.array(result[3]))))

#문서1과 문서5의 유사도 분석
print("문석1과 문서2의 유사도 : "+ str(dist(np.array(result[0]), np.array(result[4]))))

#문서1과 문서6의 유사도 분석
print("문석1과 문서2의 유사도 : "+ str(dist(np.array(result[0]), np.array(result[5]))))

print("----------------------------")

# 유사도 분석 결과를 저장하기 위해 dic객체 선언
res = {}

#문서1과 유사한 문서는?
for i in range(N): # 각 문서에 대해서 아래 명령을 수행

    # 문서번호
    doc_number = i + 1

        # 비교대상인 문서1을 제외하고 나머지 문서와 코사인 유사도 결과를 저장하기
    if doc_number != 1:
        # 유클리드 거리 유사도 결과값
        u_res = dist(np.array(result[0]), np.array(result[i]))

        # 결과값을 dic에 저장
        res[i] = u_res

        print("문서1과 문서" + str(doc_number) + "의 유사도 : " + str(u_res))

print("----------------------------")
print("결과값 : "+ str(res))
print("----------------------------")
print("문서1과 가장 유사한 문서는? : ")

# dic데이터의 value를 값에 큰 순서에 따라 정렬하기
my_doc = sorted(res.items(), key=operator.itemgetter(1), reverse=True)

# 첫번째 항목이 value 값이 가장 크기 때문에 첫번째 항목을 출력함
print("결과 : " + str(my_doc[0]))
print("문서1 : " + str(org_docs[0]))
print("문서"+ str(my_doc[0][0] + 1) + " : " + str(org_docs[my_doc[0][0]]))