aggregate_cols = {
    'ohlcv': {
        'date': 'first',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    },
    "funding": {
        'date': 'first',
        'fundingRate': 'first',
        'fundingRateDaily': 'first',
    },
    "volatility": {
        'date': 'first',
        '.BVOL': 'last',
        '.BVOL7D': 'last',
        '.BVOL24H': 'last',
        'vwap': 'mean'
    }
}

df_cols = {
    'ohlcv': [
        'date',
        'open',
        'high',
        'low',
        'close',
        'volume'
    ],
    'funding': [
        'date',
        'fundingRate',
        'fundingRateDaily'
    ],
    'volatility': [
        'date',
        '.BVOL',
        '.BVOL7D',
        '.BVOL24H',
        'vwap'
    ]
}