### EDA　###

import pandas as pd
import numpy as np
from IPython.display import Image, display
import matplotlib.pyplot as plt

# データの読み込み
train_df = pd.read_csv('../data/raw/train.csv')
test_df = pd.read_csv('../data/raw/test.csv')

# 欠損値 -1 を　 NaNに置き換える
train_df[['price_am', 'price_pm']] = train_df[['price_am', 'price_pm']].replace(-1, np.nan)
test_df[['price_am', 'price_pm']] = test_df[['price_am', 'price_pm']].replace(-1, np.nan)

# 訓練データとテストデータをマージ

train_df['is_train'] = 1
test_df['is_train'] = 0

train_cols = train_df.columns.tolist()
test_cols = test_df.columns.tolist()
common_cols = list(set(train_cols) & set(test_cols))
train_only = list(set(train_cols) - set(common_cols))
test_only = list(set(test_cols) - set(common_cols))

for col in train_only:
    test_df[col] = None

for col in test_only:
    train_df[col] = None

merged_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)

merged_df['price'] = (merged_df['price_am'] + merged_df['price_pm']) / 2

pd.set_option('future.no_silent_downcasting', True)
merged_df['y'] = merged_df['y'].fillna(0).astype(int)

merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])
merged_df['year'] = merged_df['datetime'].apply(lambda x: x.year)
merged_df['month'] = merged_df['datetime'].apply(lambda x: x.month)
merged_df['weekday'] = merged_df['datetime'].apply(lambda x: x.weekday())

merged_df.set_index('datetime', inplace=True)
merged_df.index = pd.to_datetime(merged_df.index)

# 日付（datetime）　を index　にする
merged_df_copy = merged_df.copy()

# 月と曜日ごとの平均価格を計算
merged_df_m_w = merged_df_copy.groupby(['month', 'weekday'])[['price_am', 'price_pm']].mean().reset_index()

print('月と曜日毎の平均：')
print(merged_df_m_w.head(20))

# 欠損値を月と曜日ごとの平均価格で埋める関数
def fill_missing_price_am(row):
    if pd.isna(row['price_am']):
        # 条件に一致する行を取得
        return merged_df_m_w[
            (merged_df_m_w['month'] == row['month']) &
            (merged_df_m_w['weekday'] == row['weekday'])
        ]['price_am'].values[0]
    return row['price_am']

def fill_missing_price_pm(row):
    if pd.isna(row['price_pm']):
        # 条件に一致する行を取得
        return merged_df_m_w[
            (merged_df_m_w['month'] == row['month']) &
            (merged_df_m_w['weekday'] == row['weekday'])
        ]['price_pm'].values[0]
    return row['price_pm']

# 欠損値を埋める
merged_df['price_am'] = merged_df.apply(fill_missing_price_am, axis=1)
merged_df['price_pm'] = merged_df.apply(fill_missing_price_pm, axis=1)
merged_df['price'] = (merged_df['price_am'] + merged_df['price_pm']) / 2

# ラグ特徴量の追加
merged_df['y_lag_365'] = merged_df['y'].shift(365)

for lag in [1, 7, 365]:
    merged_df[f'price_am_lag_{lag}'] = merged_df['price_am'].shift(lag)

for lag in [1, 7, 365]:
    merged_df[f'price_pm_lag_{lag}'] = merged_df['price_pm'].shift(lag)


# 移動平均の追加
windows = [7, 30]
for window in windows:
    merged_df[f'price_am_moving_avg_{window}'] = merged_df['price_am'].rolling(window=window).mean()

for window in windows:
    merged_df[f'price_pm_moving_avg_{window}'] = merged_df['price_pm'].rolling(window=window).mean()