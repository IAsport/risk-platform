"""Package data — acquisition des prix, change EUR/USD, log-returns."""

from riskplatform.data.loader import (
    convert_to_eur,
    download_fx,
    download_prices,
    load_returns,
    to_log_returns,
)

__all__ = [
    "convert_to_eur",
    "download_fx",
    "download_prices",
    "load_returns",
    "to_log_returns",
]
