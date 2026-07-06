from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskplatform.volatility import GarchParams, fit_garch, forecast_variance, garch_variance
from riskplatform.volatility.garch import _negative_loglik

TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA = 5e-6, 0.08, 0.90


def _simulate_garch(
    n_obs: int,
    omega: float = TRUE_OMEGA,
    alpha: float = TRUE_ALPHA,
    beta: float = TRUE_BETA,
    seed: int = 123,
    burn: int = 500,
) -> pd.Series:
    """Simule un GARCH(1,1) gaussien, démarré à la variance de long terme."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n_obs + burn)
    sigma2 = omega / (1.0 - alpha - beta)
    values = np.empty(n_obs + burn)
    for t in range(n_obs + burn):
        values[t] = np.sqrt(sigma2) * eps[t]
        sigma2 = omega + alpha * values[t] ** 2 + beta * sigma2
    return pd.Series(
        values[burn:], index=pd.date_range("2010-01-01", periods=n_obs, freq="B")
    )


@pytest.fixture(scope="module")
def simulated() -> pd.Series:
    return _simulate_garch(3000)


@pytest.fixture(scope="module")
def fitted(simulated: pd.Series) -> GarchParams:
    return fit_garch(simulated)


def test_mle_recovers_true_parameters_on_long_series(fitted: GarchParams):
    # MLE sur 3000 points : écart-types asymptotiques ~1e-2 sur alpha/beta.
    assert fitted.alpha == pytest.approx(TRUE_ALPHA, abs=0.03)
    assert fitted.beta == pytest.approx(TRUE_BETA, abs=0.04)
    assert fitted.omega == pytest.approx(TRUE_OMEGA, rel=0.5)
    assert fitted.persistence < 1.0


def test_estimated_loglik_not_below_true_params_loglik(simulated: pd.Series, fitted: GarchParams):
    values = simulated.to_numpy()
    sigma2_init = float(np.var(values, ddof=1))
    neg_ll_true = _negative_loglik(
        np.array([TRUE_OMEGA, TRUE_ALPHA, TRUE_BETA]), values, sigma2_init
    )
    assert fitted.loglik >= -neg_ll_true - 1e-6


def test_parameters_match_arch_reference(simulated: pd.Series, fitted: GarchParams):
    """Validation croisée contre la lib arch (oracle, tolérances SPEC B1.8 #4).

    Protocole : la vraisemblance GARCH a une
    crête plate le long de alpha+beta ; le démarrage par défaut d'arch
    sous-converge (il reste à son point initial, avec une log-vraisemblance
    INFÉRIEURE à la nôtre). On vérifie donc deux choses plus fortes que la
    simple comparaison de paramètres :
      1. notre optimum vaut au moins celui d'arch par défaut (log-vraisemblance) ;
      2. arch, démarré à NOS paramètres (même initialisation de récursion via
         backcast), n'en bouge pas — nos paramètres sont aussi un maximum de
         SON objectif, aux tolérances de la spec.
    """
    arch = pytest.importorskip("arch")

    sample_variance = float(np.var(simulated.to_numpy(), ddof=1))
    model = arch.arch_model(
        simulated, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )

    default_fit = model.fit(disp="off", backcast=sample_variance)
    assert fitted.loglik >= float(default_fit.loglikelihood) - 1e-4

    reference = model.fit(
        disp="off",
        backcast=sample_variance,
        tol=1e-12,
        starting_values=np.array([fitted.omega, fitted.alpha, fitted.beta]),
    )
    ref_omega = float(reference.params["omega"])
    ref_alpha = float(reference.params["alpha[1]"])
    ref_beta = float(reference.params["beta[1]"])

    assert fitted.alpha == pytest.approx(ref_alpha, abs=1e-3)
    assert fitted.beta == pytest.approx(ref_beta, abs=1e-3)
    assert fitted.omega == pytest.approx(ref_omega, rel=0.05)
    ref_long_run = ref_omega / (1.0 - ref_alpha - ref_beta)
    assert fitted.long_run_variance == pytest.approx(ref_long_run, rel=0.01)


def test_long_run_variance_and_persistence_formulas():
    params = GarchParams(omega=2e-6, alpha=0.1, beta=0.85, loglik=0.0, n_obs=100)

    assert params.persistence == pytest.approx(0.95)
    assert params.long_run_variance == pytest.approx(2e-6 / 0.05)


def test_filter_follows_recursion_and_ignores_r_t(simulated: pd.Series, fitted: GarchParams):
    sigma2 = garch_variance(simulated, fitted)
    values = simulated.to_numpy()

    # sigma²_t = omega + alpha·r²_{t-1} + beta·sigma²_{t-1} : r_t n'apparaît pas.
    for t in [1, 100, len(values) - 1]:
        expected = (
            fitted.omega + fitted.alpha * values[t - 1] ** 2 + fitted.beta * sigma2.iloc[t - 1]
        )
        assert sigma2.iloc[t] == pytest.approx(expected)
    assert sigma2.index.equals(simulated.index)


def test_forecast_converges_to_long_run_variance(fitted: GarchParams):
    sigma2_next = 4.0 * fitted.long_run_variance  # départ en régime stressé

    forecast = forecast_variance(fitted, sigma2_next, horizon=5000)

    assert forecast[0] == pytest.approx(sigma2_next)
    assert forecast[-1] == pytest.approx(fitted.long_run_variance, rel=1e-6)
    assert np.all(np.diff(forecast) <= 1e-18)  # décroissance monotone vers sigma²_LT


def test_forecast_is_flat_when_starting_at_long_run(fitted: GarchParams):
    forecast = forecast_variance(fitted, fitted.long_run_variance, horizon=10)

    np.testing.assert_allclose(forecast, fitted.long_run_variance)


def test_forecast_second_step_hand_computed():
    params = GarchParams(omega=2e-6, alpha=0.1, beta=0.85, loglik=0.0, n_obs=100)
    sigma2_next = 1e-4

    forecast = forecast_variance(params, sigma2_next, horizon=2)

    long_run = params.long_run_variance
    assert forecast[1] == pytest.approx(long_run + 0.95 * (sigma2_next - long_run))


def test_constant_series_rejected():
    returns = pd.Series([0.01] * 300, index=pd.date_range("2020-01-01", periods=300))

    with pytest.raises(ValueError, match="constant"):
        fit_garch(returns)


def test_short_series_rejected():
    returns = _simulate_garch(100)

    with pytest.raises(ValueError, match="min_obs"):
        fit_garch(returns)


def test_nan_rejected():
    returns = _simulate_garch(300)
    returns.iloc[10] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        fit_garch(returns)


def test_forecast_invalid_inputs_rejected(fitted: GarchParams):
    with pytest.raises(ValueError, match="horizon"):
        forecast_variance(fitted, 1e-4, horizon=0)
    with pytest.raises(ValueError, match="sigma2_next"):
        forecast_variance(fitted, -1e-4, horizon=5)
