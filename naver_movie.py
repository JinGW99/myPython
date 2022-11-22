import pandas as pd
import numpy as np
import pickle
from konlpy.tag import Okt
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 학습용 데이터
train_data = pd.read_table("data/ratings_train.txt")

# 생성할 학습 모델의 평가를 테스트할 데이터
test_data = pd.read_table("data/ratings_test.txt")

print("학습용 데이터의 네이버영화 리뷰 개수 : ", len(train_data))
print("테스트 데이터의 네이버영화 리뷰 개수 : ", len(test_data))

# 데이터가 정상적으로 가져오는지 확인하기 위해 상위 10개 출력
print(train_data[:10])
print(test_data[:10])

###############################################################
# 학습용 데이터 정제 시작
###############################################################

# document의 리뷰 내용과 label의 긍정, 부정 레코드의 중복이 존재하는지 확인
print("중복 제거된 학습용 데이터 수 확인 : ", train_data["document"].nunique(), train_data["label"].nunique())

# document의 리뷰 중복인 내용이 있다면 중복 제거
train_data.drop_duplicates(subset=["document"], inplace=True)

print("중복 제거된 최종 학습용 데이터 수 : ", len(train_data))

# 라벨 값들의 리뷰의 수 확인
print(train_data.groupby("label").size().reset_index(name = "count"))

# 널(Null)값이 존재하는 학습용 데이터 확인
print("널(Null)값이 존재하는 학습용 데이터 확인 : ", train_data.isnull().values.any())

print("널(Null)값이 존재하는 학습용 데이터 수")
print(train_data.isnull().sum())

print("널(Null)값인 데이터 확인")
print(train_data.loc[train_data.document.isnull()])

# 널(Null)값 제거
train_data = train_data.dropna(how = "any")
print("널(Null)값이 존재하는 학습용 데이터 다시 확인 : ", train_data.isnull().values.any())

print("널(Null)값이 제거된 최종 학습용 데이터 수 : ", len(train_data))

# 한글과 공백을 제외하고 모두 제거
train_data["document"] = train_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 공백을 빈 값으로 변경
train_data["document"] = train_data["document"].str.replace('^ +', "")

# 빈 값 있는지 체크
print(train_data["document"].replace("", np.nan, inplace=True))

print("널(Null)값이 존재하는 학습용 데이터 다시 확인")
print(train_data.isnull().sum())

# 널(Null)값 제거
train_data = train_data.dropna(how = 'any')
print("한글 외 단어 및 널(Null)값이 제거된 최종 학습용 데이터 수 :", len(train_data))

###############################################################
# 테스트용 데이터 정제 시작
###############################################################

# document의 리뷰 내용과 label의 긍정, 부정 레코드의 중복이 존재하는지 확인
print("중복 제거된 테스트용 데이터 수 확인 : ", test_data["document"].nunique(), test_data["label"].nunique())

# document의 리뷰 중복인 내용이 있다면 중복 제거
test_data.drop_duplicates(subset=["document"], inplace=True)

print("중복 제거된 최종 테스트용용 데이터 수 : ", len(test_data))

#train_data['label'].value_counts().plot(kind = 'bar')

# 라벨 값들의 리뷰의 수 확인
print(test_data.groupby("label").size().reset_index(name = "count"))

# 널(Null)값이 존재하는 테스트용 데이터 확인
print("널(Null)값이 존재하는 테스트용 데이터 확인 : ", test_data.isnull().values.any())

print("널(Null)값이 존재하는 테스트용 데이터 수")
print(test_data.isnull().sum())

print("널(Null)값인 데이터 확인")
print(test_data.loc[test_data.document.isnull()])

# 널(Null)값 제거
test_data = test_data.dropna(how = "any")
print("널(Null)값이 존재하는 테스트용 데이터 다시 확인 : ", test_data.isnull().values.any())

print("널(Null)값이 제거된 최종 테스트용 데이터 수 : ", len(test_data))

# 한글과 공백을 제외하고 모두 제거
test_data["document"] = test_data["document"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# 공백을 빈 값으로 변경
test_data["document"] = test_data["document"].str.replace('^ +', "")

# 빈 값 있는지 체크
print(test_data["document"].replace("", np.nan, inplace=True))

print("널(Null)값이 존재하는 테스트용 데이터 다시 확인")
print(test_data.isnull().sum())

# 널(Null)값 제거
test_data = test_data.dropna(how = 'any')
print("한글 외 단어 및 널(Null)값이 제거된 최종 테스트용 데이터 수 :", len(test_data))

