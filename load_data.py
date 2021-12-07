'''
Author: Sumin Lim @ KAIST
Date: 2021-10-25
Usage: python load_data.py
Description:
    * Preprocessing of driver's log data
    * Generate tensor
    * Split data for training and test
'''
from typing import List, Tuple
from tqdm import tqdm
from datetime import datetime
import pickle as pkl
import numpy as np
import pandas as pd
import h3

def load_data(filepath: str='../data/07_03.csv') -> pd.DataFrame:
    df = pd.read_csv(filepath, low_memory=False)
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

def get_time_interval(df: pd.DataFrame, time_interval: str='24H') -> List:
    if time_interval == '24H':
        dates = [datetime.strptime(x, '%Y-%m-%d') for x in
                    df['pickup_date'].unique().tolist()]
    else:
        dates_start = df['pickup_dt'].min()
        dates_end = df['pickup_dt'].max()

        if dates_start.minute < 30:
            dates_start = dates_start.replace(minute=0, second=0)
        else:
            dates_start = dates_start.replace(minute=30, second=0)

        if dates_end.minute < 30:
            dates_end = dates_end.replace(minute=0, second=0)
        else:
            dates_end = dates_end.replace(minute=30, second=0)

        dates = pd.date_range(start=dates_start,
                              end=dates_end,
                              freq=time_interval,
                              closed=None)

        if df['pickup_dt'].max() > dates[-1]:
            time_add = int(''.join([x for x in time_interval if x.isdigit()]))
            time_unit = ''.join([x for x in time_interval if x.isalpha()])
            time_lastslot = dates[-1]+pd.Timedelta(time_add, unit=time_unit)
            dates = dates.union([time_lastslot])

    return dates

def construct_matrix(time_interval='24H'):
    df = load_data()

    rows, cols = zip(*df['h3_xy'].unique())
    rows_range = sorted(set(rows))
    cols_range = sorted(set(cols))

    rows_scaled = {x: idx for idx, x in enumerate(rows_range)}
    cols_scaled = {x: idx for idx, x in enumerate(cols_range)}

    locations = df['h3_xy_scale'].unique().tolist()
    dates = get_time_interval(df, time_interval)

    tensor = {}
    tensor_binary = {}
    n_rows, n_cols = len(rows_scaled.values()), len(cols_scaled.values())

    print('\nMaking matrices with time interval...\n')
    for idx, date in tqdm(enumerate(dates), total=len(dates)):
        if time_interval == '24H':
            temp = df[df['pickup_date'] == date.strftime('%Y-%m-%d')]
        else:
            #temp = df[dates[idx] < df['pickup_dt'] < dates[idx+1]]
            try:
                temp = df[(df['pickup_dt'] < dates[idx+1])
                        & (df['pickup_dt'] >= dates[idx])]
            except IndexError:
                break

        demand_cnt = temp.groupby('h3_xy_scale')['numbering'].count()
        if demand_cnt.shape[0] == 0:
            continue
        temp_rows, temp_cols = zip(*list(demand_cnt.index))

        data = np.zeros((n_rows, n_cols))
        data[temp_rows, temp_cols] = demand_cnt
        tensor[date] = data

        #data_binary = np.zeros((n_rows, n_cols))
        #data_binary[temp_rows, temp_cols] = 1
        #tensor_binary[date] = data_binary

    #sliding_dates_week = [dates[i:i+7] for i in range(len(dates)-6)]

    return dates, tensor
    #return sliding_dates_week, tensor, tensor_binary

def main(time_interval: str,
        n_sliding: int,
        sliding: bool=True,
        is_saved: bool=True) -> Tuple[np.array]:

    if is_saved and sliding:
        X_train = pkl.load(
                open(f'data/X_train_{time_interval}_{n_sliding}.pkl', 'rb'))
        X_test = pkl.load(
                open(f'data/X_test_{time_interval}_{n_sliding}.pkl', 'rb'))
        y_train = pkl.load(
                open(f'data/y_train_{time_interval}_{n_sliding}.pkl', 'rb'))
        y_test = pkl.load(
                open(f'data/y_test_{time_interval}_{n_sliding}.pkl', 'rb'))
        return X_train, X_test, y_train, y_test

    elif is_saved and (not sliding):
        tensor = pkl.load(open(f'data/tensor_{time_interval}.pkl', 'rb'))
        return np.array(list(tensor.values()))

    #sliding_dates_week, tensor, _ = main_data()
    _, tensor = main_data(time_interval=time_interval)
    dates = list(tensor.keys())
    sliding_dates = [dates[i:i+n_sliding] for i in range(len(dates)-n_sliding)]
    n_weeks_sliding_train = (len(sliding_dates) * 2) // 3
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for idx in range(len(sliding_dates)):
        slot = sliding_dates[idx]
        if sliding:
            X_data = np.array([np.expand_dims(tensor[x],axis=-1)
                for x in slot[:n_sliding]])
            y_data = np.array([np.expand_dims(tensor[slot[-1]],axis=-1)])
            if idx < n_weeks_sliding_train:
                X_train.append(X_data)
                y_train.append(y_data)
            else:
                X_test.append(X_data)
                y_test.append(y_data)
        else:
            return np.array(list(tensor.values()))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_train = np.swapaxes(y_train, 1, -1)
    y_test = np.swapaxes(y_test, 1, -1)

    pkl.dump(X_train,
            open(f'data/X_train_{time_interval}_{n_sliding}.pkl', 'wb'))
    pkl.dump(X_test,
            open(f'data/X_test_{time_interval}_{n_sliding}.pkl', 'wb'))
    pkl.dump(y_train,
            open(f'data/y_train_{time_interval}_{n_sliding}.pkl', 'wb'))
    pkl.dump(y_test,
            open(f'data/y_test_{time_interval}_{n_sliding}.pkl', 'wb'))

    return X_train, X_test, y_train, y_test

if __name__=='__main__':
    X_train, X_test, y_train, y_test = main('30min', 4)
