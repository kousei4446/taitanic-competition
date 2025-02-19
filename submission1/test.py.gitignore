import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データの読み込み
train_df = pd.read_csv("../data/train.csv")

# 必要な特徴量の選択
X = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Cavin","Embarked"]]

print(X)


# 欠損値の処理,AgeとFareに欠損値がある場合、その列の平均値で欠損値を埋めます。
X["Age"].fillna(X["Age"].mean(), inplace=True)
X["Fare"].fillna(X["Fare"].mean(), inplace=True)
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)

print("--------------------------------------------------------------------------------------------------------------------")
print(X.head())  # 最初の5行だけ表示
print(X.columns)  # 列名のリスト



