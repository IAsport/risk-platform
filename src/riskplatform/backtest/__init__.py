"""Package backtest — exceptions, Kupiec POF, Christoffersen tests.

Backtesting compares out-of-sample VaR forecasts with realized losses.
An exception occurs when realized loss L_t = -r_t * notional is strictly greater
than the VaR forecast for the same date.

See ARCHITECTURE.md and SPEC.md section 6.
"""

from riskplatform.backtest.christoffersen import christoffersen_cc, christoffersen_independence
from riskplatform.backtest.es_backtest import acerbi_szekely_z2
from riskplatform.backtest.exceptions import count_exceptions
from riskplatform.backtest.kupiec import kupiec_pof
from riskplatform.backtest.traffic_light import (
    basel_zone_bounds,
    rolling_traffic_light,
    traffic_light,
)

__all__ = [
    "acerbi_szekely_z2",
    "basel_zone_bounds",
    "christoffersen_cc",
    "christoffersen_independence",
    "count_exceptions",
    "kupiec_pof",
    "rolling_traffic_light",
    "traffic_light",
]
