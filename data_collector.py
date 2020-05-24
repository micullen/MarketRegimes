from typing import List, Optional, Dict, Any
from datetime import datetime
from pandas import DataFrame

import arrow
import jsonschema
import pandas as pd

import ccxt
from tenacity import retry, retry_if_exception_type, wait_exponential, stop_after_attempt, before_log, RetryError

from MarketClusters.exchange import *


logger = logging.getLogger()

class DataProvider:
    """This class accepts an exchange and a pair then periodically updates market data (OHLCV)."""

    def __init__(self, exchange: Exchange, aggregate_cols, df_cols, args=None) -> None:
        self._cache: List[List[Any]] = []
        self.exchange: Exchange = exchange
        self.aggregate_cols = aggregate_cols
        self.df_cols = df_cols
        self.args = args

    @retry(retry=retry_if_exception_type((ccxt.ExchangeError, ccxt.NetworkError)),
           wait=wait_exponential(multiplier=5, min=1, max=60),
           stop=stop_after_attempt(5),
           before_sleep=before_log(logger, logging.INFO))
    def _fetch_ohlcv(self, pair: str, timeframe: str, since: int, limit: int) -> List[List[Any]]:
        return self.exchange.ccxt.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)

    @retry(retry=retry_if_exception_type((ccxt.ExchangeError, ccxt.NetworkError)),
           wait=wait_exponential(multiplier=5, min=1, max=60),
           stop=stop_after_attempt(5),
           before_sleep=before_log(logger, logging.INFO))
    def _fetch_volatility(self, symbol: str, timeframe: str, since: int, limit: int) -> List:
        x = self.exchange.ccxt.public_get_trade_bucketed(
            {"symbol": symbol, 'reverse': False, 'count': str(limit), "binSize": timeframe,
             "partial": True, "startTime": str(datetime.fromtimestamp(since / 1000))})
        return x

    @retry(retry=retry_if_exception_type((ccxt.ExchangeError, ccxt.NetworkError)),
           wait=wait_exponential(multiplier=5, min=1, max=60),
           stop=stop_after_attempt(5),
           before_sleep=before_log(logger, logging.INFO))
    def _fetch_bitmex_funding(self, symbol: str, since: int, limit: int) -> List:
        return self.exchange.ccxt.public_get_funding({"symbol": 'XBT', 'reverse': False, 'count': str(limit),
                                                 "startTime": str(datetime.fromtimestamp(since / 1000))})

    @retry(retry=retry_if_exception_type((ccxt.ExchangeError, ccxt.NetworkError)),
           wait=wait_exponential(multiplier=5, min=1, max=60),
           stop=stop_after_attempt(5),
           before_sleep=before_log(logger, logging.INFO))
    def _fetch_bitmex_vwap(self, symbol: str, since: int, limit: int) -> List:
        return self.exchange.ccxt.public_get_trade_bucketed({"symbol": 'XBT', 'reverse': False, 'count': str(limit),
                                                 "startTime": str(datetime.fromtimestamp(since / 1000))})

    def _bitmex_data_massager(self, response: Dict, *args: str, vwap=False) -> List:
        """"""
        metric_list = []
        for interval in response:
            reduced_dict = {}
            reduced_dict['timestamp'] = int(datetime.timestamp(
                datetime.strptime(interval['timestamp'], '%Y-%m-%dT%H:%M:%S.%fZ')) * 1000)
            for arg in args:
                reduced_dict[str(arg)] = interval[str(arg)]

            reduced_list = list(reduced_dict.values())
            metric_list.append(reduced_list)

        return metric_list

    def _data_fetcher(self, pair: str, tick_interval: str, since_ms: Optional[int] = None,
                      num_candles: int = 5000) -> Optional[List[List[Any]]]:
        interval_seconds = self.exchange.ccxt.parse_timeframe(tick_interval)
        interval_mins = interval_seconds / 60

        # Last item should be in the time interval [now - tick_interval, now]
        until_ms = arrow.utcnow().shift(
            minutes=-interval_mins
        ).timestamp * 1000

        # Initialising a starting time period to send to api if it does not exist
        if not since_ms:
            since_ms = arrow.utcnow().shift(
                minutes=-(interval_mins * num_candles)
            ).timestamp * 1000

        lim = min(num_candles, 500)

        # Some exchanges don't have USDT ticker data, so get USD instead, which should be the same
        if pair.endswith('USDT'):
            pair = pair.replace('USDT', 'USD')
        data_part = 0
        data: List[List[Any]] = []
        while not since_ms or since_ms < until_ms:
            try:
                if self.function == '_fetch_ohlcv':
                    data_part = self._fetch_ohlcv(pair, timeframe=tick_interval, since=since_ms, limit=lim)

                elif self.function == '_fetch_volatility_vwap':
                    vol_symbols = ['.BVOL', '.BVOL7D', '.BVOL24H', 'XBT']
                    vols = []
                    for symbol in vol_symbols:
                        response = self._fetch_volatility(symbol, timeframe=tick_interval, since=since_ms, limit=lim)
                        if symbol == 'XBT':
                            arg = 'vwap'
                        else:
                            arg = 'high'
                        data_part = self._bitmex_data_massager(response, arg)
                        vols.append(data_part)

                    # Append 2nd, 3rd, 4th volatility list values to the first
                    vols_data = vols[0]
                    for i in range(len(vols_data)):
                        vols_data[i].append(vols[1][i][1])
                        vols_data[i].append(vols[2][i][1])
                        vols_data[i].append(vols[3][i][1])

                    data_part = vols_data

                elif self.function == '_fetch_bitmex_funding':
                    response = self._fetch_bitmex_funding(pair, since=since_ms, limit=lim)
                    data_part = self._bitmex_data_massager(response, 'fundingRate', 'fundingRateDaily')

            except RetryError as e:
                logger.error(f'Failed to fetch {lim} candles from {since_ms}. Message: {e}')
                return None

            # Ensure the data is in ascending time order
            data_part = sorted(data_part, key=lambda x: x[0])

            if not data_part:
                logger.warning('Data returned by ccxt is empty. Something weird has happened.')
                break

            logger.debug('Downloaded data for %s time range [%s, %s]',
                         pair,
                         arrow.get(data_part[0][0] / 1000).format(),
                         arrow.get(data_part[-1][0] / 1000).format())

            data.extend(data_part)
            logger.debug(f'Paginator progress: {len(data)} / {num_candles}')
            # The next data we request should be directly after the last candle we have just received, so add the
            # interval to the timestamp of the last candle so we can fetch from there on the next loop.
            since_ms = data[-1][0] + (interval_seconds * 1000)
        return data

    def _fill_cache(self, pair: str, interval: str) -> None:
        logger.debug('Populating cache for %s-%s', pair, interval)
        data = self._data_fetcher(pair, interval, num_candles=20000)
        if data is None:
            return
        # We explicitly WANT to include the last unfinished candle, so we don't drop the last row here
        self._cache = data

    def _update_cache(self, pair: str, interval: str) -> None:
        if len(self._cache) == 0:
            # The cache is empty, so let's fill it
            logger.debug('Filling empty cache')
            self._fill_cache(pair, interval)
        else:
            # The cache is already filled, so lets just fetch the latest few candles and add to it
            candles = self._data_fetcher(pair, interval, num_candles=2)
            if candles is None:
                logger.warning(f'Updating cache failed - None returned.')
                return
            elif len(candles) < 2:
                logger.warning(f'Updating cache failed - {len(candles)} returned.')
                return

            # The idea here is that we pull the last two candles and if they are two candles we already have then we
            # just replace them. The second most recent candle should be a final/closed candle, and the most recent
            # one should be an unfinished candle. If the second most recent candle we just received doesn't match the
            # second most recent we have stored in our cache then this is means that a new candle has opened and we
            # now have the final version of the previously unfinished candle. Therefore instead of replacing the last
            # two we just remove the last one and add a new one.
            if candles[0][0] > self._cache[-2][0]:
                # We've got a new candle so remove one from the start of the cache
                del self._cache[0]
                self._cache = self._cache[:-1] + candles
            else:
                self._cache = self._cache[:-2] + candles

    def _get_candles_and_update_cache(self, pair: str, interval: str) -> List[List[Any]]:
        self._update_cache(pair, interval)
        return self._cache

    def _resample_to_intervals(self, data: DataFrame, intervals: List[str]) -> Dict[str, DataFrame]:
        """
        Resamples the given dataframe to the desired intervals
        :param data: A DataFrame containing OHLCV data
        :param intervals: A list of intervals in string format which we want to resample to
        :return: A dictionary of intervals to dataframes containing data resampled to that interval
        """
        df_dict = {"base": data}

        additional_df = data.copy()
        additional_df = additional_df.set_index(pd.DatetimeIndex(additional_df['date']))
        if self.function == '_fetch_ohlcv':
            aggregate_cols = self.aggregate_cols['ohlcv']

        elif self.function == '_fetch_volatility_vwap':
            aggregate_cols = self.aggregate_cols['volatility']

        elif self.function == '_fetch_bitmex_funding':
            aggregate_cols = self.aggregate_cols['funding']


        # For each interval resample and then add to dictionary of dataframes
        for interval in intervals:
            # Change from string representation, e.g. '1h' to integer number of minutes, e.g. 60
            interval_mins = self.exchange.ccxt.parse_timeframe(interval) / 60
            resampled = additional_df.copy()
            resampled.set_index(pd.DatetimeIndex(resampled['date']), inplace=True)
            resampled = resampled.resample(f'{interval_mins}Min', label='left').agg(aggregate_cols)
            resampled['date'] = resampled.index
            resampled = resampled.reset_index(drop=True)

            df_dict[interval] = resampled

        return df_dict

    def get_latest_dataframes(self, pair: str, function, interval: str,
                              additional_intervals: Optional[List[str]] = None) -> Optional[Dict[str, DataFrame]]:
        # Delete cache from previous loop
        self._cache = []
        self.function = function
        raw_data = self._get_candles_and_update_cache(pair, interval)
        if not raw_data:
            return None

        if self.function == '_fetch_ohlcv':
            df_cols = self.df_cols['ohlcv']

        elif self.function == '_fetch_volatility_vwap':
            df_cols = self.df_cols['volatility']

        elif self.function == '_fetch_bitmex_funding':
            df_cols = self.df_cols['funding']

        df = pd.DataFrame(raw_data, columns=df_cols)

        # Convert the date column from millisecond Unix timestamps to Pandas DateTimes
        df.date = pd.to_datetime(df.date, unit='ms')

        # In case there are no additional intervals in the config, just use base dataframe
        if additional_intervals:
            df_dict = self._resample_to_intervals(df, additional_intervals)
        else:
            df_dict = {"base": df}

        return df_dict