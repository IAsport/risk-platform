"""Package var — VaR historique, paramétrique, Monte Carlo, scaling, rolling.

SIGN CONVENTION: every public function returns a POSITIVE LOSS. Internally,
losses are L = -r_p, and VaR_alpha is the alpha-quantile of losses, equivalently
the 1-alpha quantile of portfolio returns with the sign reversed.

See ARCHITECTURE.md and SPEC.md sections 3-4.
"""

from riskplatform.var.conditional import (
    rolling_var_conditional,
    var_conditional,
    var_conditional_monte_carlo,
)
from riskplatform.var.historical import var_historical
from riskplatform.var.monte_carlo import var_monte_carlo, var_monte_carlo_student
from riskplatform.var.parametric import var_parametric, var_parametric_portfolio
from riskplatform.var.rolling import rolling_var, scale_var

__all__ = [
    "rolling_var",
    "rolling_var_conditional",
    "scale_var",
    "var_conditional",
    "var_conditional_monte_carlo",
    "var_historical",
    "var_monte_carlo",
    "var_monte_carlo_student",
    "var_parametric",
    "var_parametric_portfolio",
]
