# %%
import os

import numpy as np
import pandas as pd

random_seed = 1
np.random.seed(random_seed)

# %%
stations_df = pd.read_csv('./data/stations.csv')
stations_dict = stations_df.groupby(['common_id']).first().to_dict('index')
stations_dict['2386-ch']['lower_limit'] = 38_000.0
stations_dict['2386-ch']['upper_limit'] = 40_000.0
stations_dict['2720050000-de']['lower_limit'] = 20.0
stations_dict['2720050000-de']['upper_limit'] = 500.0
stations_dict['36022-ie']['lower_limit'] = 20.0
stations_dict['36022-ie']['upper_limit'] = 175.0
stations_dict['39003-ie']['lower_limit'] = 10.0
stations_dict['39003-ie']['upper_limit'] = 225.0
stations_dict['42960105-de']['lower_limit'] = -10.0
stations_dict['42960105-de']['upper_limit'] = 275.0
stations_dict['auto-1003803']['lower_limit'] = -10.0
stations_dict['auto-1003803']['upper_limit'] = 275.0

# %%
for common_id, station_dict in stations_dict.items():
    fp = f'./data/classified_raw/{common_id}_outliers_classified.parquet'
    if not os.path.exists(fp):
        continue
    raw_classified_df = pd.read_parquet(fp)
    print(f'Removing for {common_id}, lower limit '
          f'{station_dict["lower_limit"]}, upper limit '
          f'{station_dict["upper_limit"]}')
    print('Removing following rows:')
    mask = (raw_classified_df['water_level'] < station_dict['lower_limit']) | (
            raw_classified_df['water_level'] > station_dict['upper_limit'])
    print(raw_classified_df[mask])
    raw_classified_df[~mask].to_parquet(
        f'././data/classified/{common_id}_outliers_classified.parquet')
    print()
