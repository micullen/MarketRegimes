# MarketRegimes

This repository retrieves market features, such as Volatility, funding and candle data from an exchange API, then creates Market Regimes based on these features using a K-means clustering algorithm. The code heavily depends upon pandas for data sorting and processing.


## Usage

```python
python __main__.py

```
The amount of data trained on needs to be altered to your choosing within the ``CreateModel class``. Following this, you can choose to plot in-sample data or out-of-sample data within ``run_model``.

## Requirements
pandas
ccxt
sklearn

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
