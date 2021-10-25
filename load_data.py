'''
Author: Sumin Lim @ KAIST
Date: 2021-10-25
Usage: python load_data.py
Description:
    * Preprocessing of driver's log data
    * Generate tensor
    * Split data for training and test
'''
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import h3

def load_data(filepath: str='../data/07_03.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.reset_index()
    df.rename(columns={'index': 'numbering'}, inplace=True)

    df['request_dt'] = pd.to_datetime(df['request_at'].str.rstrip('UTC'),
            format='%Y-%m-%d %H:%M:%S')
    df['pickup'] = df['pickup_date'] + ' ' + df['pickup_time']
    df['pickup_dt'] = pd.to_datetime(df['pickup'], format='%Y-%m-%d %H:%M:%S')

    days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday']
    df['pickup_weekday'] = pd.Categorical(df['pickup_weekday'],
                                          categories=days,
                                          ordered=True)

    lat_cond = ((df['pickup_latitude'] <= 38.2033)
             & (df['pickup_latitude'] >= 37.1897))
    lng_cond = ((df['pickup_longitude'] <= -121.5871)
             & (df['pickup_longitude'] >= -122.6445))
    df = df[lat_cond & lng_cond]
    df['pickup_loc'] = tuple(zip(df['pickup_latitude'], df['pickup_longitude']))

    bay_center = [37.8272, -122.2913]
    bay_center_cell = h3.geo_to_h3(bay_center[0], bay_center[1], resolution=7)
    df['h3_7'] = df['pickup_loc'].apply(lambda x:
                                       h3.geo_to_h3(x[0], x[1], resolution=7))

    df['h3_xy'] = df['h3_7'].apply(
        lambda x: h3.experimental_h3_to_local_ij(bay_center_cell, x))

    rows, cols = zip(*df['h3_xy'].unique())
    rows_range = sorted(set(rows))
    cols_range = sorted(set(cols))

    rows_scaled = {x: idx for idx, x in enumerate(rows_range)}
    cols_scaled = {y: idy for idy, y in enumerate(cols_range)}

    df['h3_xy_scale'] = df['h3_xy'].apply(
        lambda x: (rows_scaled[x[0]], cols_scaled[x[1]]))

    return df

def main():
    df = load_data()

    rows, cols = zip(*df['h3_xy'].unique())
    rows_range = sorted(set(rows))
    cols_range = sorted(set(cols))

    rows_scaled = {x: idx for idx, x in enumerate(rows_range)}
    cols_scaled = {x: idx for idx, x in enumerate(cols_range)}

    locations = df['h3_xy_scale'].unique().tolist()
    dates = [datetime.strptime(x, '%Y-%m-%d') for x in
                df['pickup_date'].unique().tolist()]
    dates = sorted(dates)

    tensor = {}
    tensor_binary = {}
    n_rows, n_cols = len(rows_scaled.values()), len(cols_scaled.values())

    for date in tqdm(dates):
        temp = df[df['pickup_date'] == date.strftime('%Y-%m-%d')]
        demand_cnt = temp.groupby('h3_xy_scale')['numbering'].count()
        temp_rows, temp_cols = zip(*list(demand_cnt.index))

        data = np.zeros((n_rows, n_cols))
        data[temp_rows, temp_cols] = demand_cnt
        tensor[date] = data

        data_binary = np.zeros((n_rows, n_cols))
        data_binary[temp_rows, temp_cols] = 1
        tensor_binary[date] = data_binary

    sliding_dates_week = [dates[i:i+7] for i in range(len(dates)-6)]

    return sliding_dates_week, tensor, tensor_binary

if __name__=='__main__':
    sliding_dates_week, tensor, tensor_binary = main()
    print(tensor)
    print(tensor_binary)
