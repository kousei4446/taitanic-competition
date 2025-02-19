import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# データの読み込み
train_df = pd.read_csv("../data/train.csv")

# 必要な特徴量の選択
X = train_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Embarked"]]

# 欠損値の処理
X["Age"].fillna(X["Age"].mean(), inplace=True)
X["Fare"].fillna(X["Fare"].mean(), inplace=True)


# 性別のダミー変数化
X = pd.get_dummies(X, columns=["Sex","Embarked"], drop_first=True)

# 目的変数
y = train_df["Survived"]

# 特徴量の標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# データの分割
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# モデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # 入力層
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 2クラス分類 -> sigmoid
])

# モデルのコンパイル
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),callbacks=[early_stopping] )

# 評価
test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
print(f"Validation accuracy: {test_acc * 100:.2f}%")

# 予測
predictions = model.predict(X_val)
predictions = (predictions > 0.5).astype(int)

# 提出用の予測結果を作成
test_df = pd.read_csv("../data/test.csv")
X_test = test_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","Embarked"]]
X_test["Age"].fillna(X_test["Age"].mean(), inplace=True)
X_test["Fare"].fillna(X_test["Fare"].mean(), inplace=True)
X_test = pd.get_dummies(X_test, columns=["Sex","Embarked"], drop_first=True)
X_test_scaled = scaler.transform(X_test)

test_predictions = model.predict(X_test_scaled)
test_predictions = (test_predictions > 0.5).astype(int)

# 提出用のCSVファイルを作成
submit = test_df[["PassengerId"]]
submit["Survived"] = test_predictions

submit.to_csv("./result.csv", index=False)
    
# 訓練データに対する精度を表示
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
print(f"Training accuracy: {train_acc * 100:.2f}%")

# バリデーションデータに対する精度を表示（すでに評価済み）
print(f"Validation accuracy: {test_acc * 100:.2f}%")