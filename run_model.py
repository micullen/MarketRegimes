from MarketClusters.data_collector import DataProvider
from MarketClusters.model_constant import aggregate_cols, df_cols
from MarketClusters.create_model import CreateModel

import exchange
import pandas as pd
import logging
import numpy as np
import json

logger = logging.getLogger()

"""
To create a model trying to encapsulate different market regimes, enter the following variables into CreateModel
in run() below. These regimes are plotted against xbtusd.
no_components - How many components to run the k-means algorithm over. If you have more features than the value
                you select, PCA will dimensionally reduce them and normalise them.
no_clusters   - This determines how many different types of market 'regimes' will be outputted.
metrics       - Full list ['.BVOL', '.BVOL7D', '.BVOL24H', 'vwap_1', 'vwap_2', 'vwap_3', 'fundingRate',
                           'fundingRateDaily', 'rets2'...'rets20', 'ma2', 'ma2_s', 'ma3', 'ma3_s']. 
                Further metrics can either be derived in process_data_merge or retrieved in data_collecter.
merged_df     - This is the full dataframe containing all the above metrics.
"""


def initialise_exchange():
    with open('config.json') as file:
        config = json.load(file)

    exchange_value = exchange.Bitmex('Bitmex', config)
    return exchange_value


def process_data_merge(data_provider):
    # Prep ohlcv and derived data for merge
    data_dict = data_provider.get_latest_dataframes('BTC/USD', '_fetch_ohlcv', '1h', ['1d'])
    data_1d = data_dict['1d']

    data_1d['rets1'] = np.log(data_1d['close']) - np.log(data_1d['close'].shift(1))
    data_1d['rets2'] = np.log(data_1d['close']) - np.log(data_1d['close'].shift(2))
    data_1d['rets5'] = np.log(data_1d['close']) - np.log(data_1d['close'].shift(5))
    data_1d['rets7'] = np.log(data_1d['close']) - np.log(data_1d['close'].shift(7))
    data_1d['rets10'] = np.log(data_1d['close'] - np.log(data_1d['close'].shift(10)))
    data_1d['rets15'] = np.log(data_1d['close'] - np.log(data_1d['close'].shift(15)))
    data_1d['rets20'] = np.log(data_1d['close']) - np.log(data_1d['close'].shift(20))

    # Prep volatility data for merge
    vol = data_provider.get_latest_dataframes('BTC/USD', '_fetch_volatility_vwap', '1h', ['1d'])
    vol = vol['1d']


    # Prep funding data for merge
    funding = data_provider.get_latest_dataframes('BTC/USD', '_fetch_bitmex_funding', '1d')
    funding = funding['base']
    # funding['date'] = funding['date'].dt.ceil(freq='24H')
    # funding = funding.set_index('date', drop=True)
    # funding = funding.resample('1D').bfill()
    # funding = funding.reset_index(drop=False)

    # Do the join
    merged = pd.merge_asof(funding, vol,
                           left_on='date',
                           right_on='date',
                           direction='backward'
                           )

    merged2 = pd.merge_asof(merged, data_1d,
                            left_on='date',
                            right_on='date',
                            direction='backward'
                            )

    return merged2


def run():
    exchange = initialise_exchange()
    data_provider = DataProvider(exchange, aggregate_cols, df_cols)

    # Merged_df contains _all_ metrics
    merged_df = process_data_merge(data_provider)
    merged_df = merged_df.set_index('date', drop=True).dropna()

    # Create an instance of the model with returns, funding, volatility all features
    vol = CreateModel(n_components=5, n_clusters=5, metrics=['.BVOL7D', '.BVOL24H', 'fundingRate', 'fundingRateDaily', 'rets1', 'rets2', 'rets5', 'rets7', 'rets10'],
                      dataframe=merged_df)

    # trend = CreateModel(n_components=4, n_clusters=4, metrics=['rets5', 'rets7'],
    #                     dataframe=merged_df)

    # Create an instance of the model with just returns as features
    # trend2 = CreateModel(n_components=4, n_clusters=4, metrics=['rets1', 'rets2', 'rets5', 'rets7', 'rets10', 'rets15', 'rets20'],
    #                      dataframe=merged_df)

    vol_model = vol.handle_model()
    # trend_model = trend.handle_model()
    # trend2_model = trend2.handle_model()

    # Pass whole trimmed data set to sort it ready for testing against model
    test_data_vol = vol.prep_data(vol.trim_df)
    # test_data_trend = trend.prep_data(trend.trim_df)
    # test_data_trend2 = trend2.prep_data(trend2.trim_df)

    # Plot the model tested on whole trimmed dataset
    vol.plot_kmeans_clusters(vol_model, merged_df, test_data_vol)
    # trend.plot_kmeans_clusters(trend_model, merged_df, test_data_trend)
    # trend2.plot_kmeans_clusters(trend2_model, merged_df, test_data_trend2)