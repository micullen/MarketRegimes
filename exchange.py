import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple
import ccxt


logger = logging.getLogger(__name__)


class Exchange(ABC):
    """
    This is the main exchange class which wraps a ccxt Exchange object. All communication to an exchange will be done
    through this class. To add a new exchange you should inherit from this class and implement the abstract methods.
    """

    def __init__(self, name: str, config: [str, Any]) -> None:  # type: ignore
        self.name = name
        self.config = config
        self.ccxt = self._init_ccxt(name, config)

    def _init_ccxt(self, name: str, config: Dict[str, Any]) -> ccxt.Exchange:
        """
        Initialise a ccxt Exchange object with the provided config and return it
        :param exchange_config: The config for the exchange we want to create
        :return: An instance of the ccxt Exchange class
        """
        if name.lower() not in ccxt.exchanges:
            raise RuntimeError(f'Exchange {name} is not supported by ccxt')

        # Get the ccxt Exchange class with the name we have specified
        ccxt_exchange_class = getattr(ccxt, name.lower(), None)
        if ccxt_exchange_class is None:
            raise RuntimeError(f"Couldn't get {name} attribute of ccxt")

        # Instantiate the ccxt Exchange class
        ccxt_exchange = ccxt_exchange_class({
            'apiKey': config['exchange']['api_key'],
            'secret': config['exchange']['api_secret_key'],
        })
        ccxt_exchange.load_markets()
        return ccxt_exchange


class Bitmex(Exchange):
    """This is a custom exchange class for BitMex. This is to add any additional methods which are not covered by
    the base class."""