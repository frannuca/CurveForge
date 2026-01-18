"""Gaussian Process projection utilities for financial time series.

Provides a small, well-documented API to fit a GaussianProcessRegressor to
features (DataFrame) and a target time series (pd.Series) and produce
projections for provided future features or using simple autoregressive
lag-based feature generation.

Usage:
    from gp_projection import GaussianProcessProjection, make_lag_features

    model = GaussianProcessProjection()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_future)

    # or helper to create X_future from past series
    X_future = make_lag_features(y_train, lags=5, n_steps=20)
    y_pred = model.predict(X_future)

The functions expect pandas DataFrame/Series inputs and return pandas.Series
with the same index as X_future (or a generated DatetimeIndex if using the
lag helper and the original series has a DatetimeIndex).
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple


class GaussianProcessProjection:
    """Gaussian Process regressor for time-series projections.

    Parameters
    - kernel: sklearn GaussianProcess kernel. If None, a default C*RBF + WhiteKernel
      kernel is used.
    - normalize_y: whether to z-score the target variable before fitting.
    - n_restarts_optimizer: passed to GaussianProcessRegressor for kernel
      hyperparameter optimization.
    - alpha: added to the diagonal of the kernel for numerical stability.
    """

    def __init__(
        self,
        kernel=None,
        normalize_y: bool = True,
        n_restarts_optimizer: int = 5,
        alpha: float = 1e-10,
    ):
        if kernel is None:
            kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=5.0, length_scale_bounds=(1e-2, 1e3))
            kernel += WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))

        self.kernel = kernel
        self.normalize_y = normalize_y
        self.n_restarts_optimizer = n_restarts_optimizer
        self.alpha = alpha

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler() if normalize_y else None
        self._gpr: Optional[GaussianProcessRegressor] = None
        self._feature_columns = None
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GaussianProcessProjection":
        """Fit the Gaussian Process to features X and target y.

        X: DataFrame, shape (n_samples, n_features)
        y: Series, shape (n_samples,)

        Returns self.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self._feature_columns = list(X.columns)

        X_vals = X.values.astype(float)
        Xs = self._scaler_X.fit_transform(X_vals)

        y_vals = y.values.reshape(-1, 1).astype(float)
        if self.normalize_y:
            ys = self._scaler_y.fit_transform(y_vals).ravel()
        else:
            ys = y_vals.ravel()

        self._gpr = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=False,  # we handle normalization ourselves
        )

        self._gpr.fit(Xs, ys)
        self._is_fitted = True
        return self

    def predict(self, X_future: pd.DataFrame, return_std: bool = False) -> pd.Series:
        """Predict future target values given future features.

        X_future: DataFrame with the same columns used during fit. The index of
                  X_future will be used for the returned Series index.
        return_std: if True, return a tuple (y_pred, y_std) where both are pd.Series.

        Returns pd.Series (or tuple of pd.Series if return_std=True).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict")
        if not isinstance(X_future, pd.DataFrame):
            raise TypeError("X_future must be a pandas DataFrame")

        # align columns
        if list(X_future.columns) != self._feature_columns:
            # try to reindex columns if possible
            try:
                X_future = X_future.reindex(columns=self._feature_columns)
            except Exception:
                raise ValueError("X_future must have the same feature columns as the training data")

        Xf_vals = X_future.values.astype(float)
        Xf_scaled = self._scaler_X.transform(Xf_vals)

        y_pred_scaled, y_std = self._gpr.predict(Xf_scaled, return_std=True)

        if self.normalize_y:
            y_pred = self._scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            # transform std approximately by multiplying with y scaler scale
            y_std = y_std * float(self._scaler_y.scale_)
        else:
            y_pred = y_pred_scaled

        y_series = pd.Series(y_pred, index=X_future.index, name="gp_projection")

        if return_std:
            std_series = pd.Series(y_std, index=X_future.index, name="gp_std")
            return y_series, std_series
        return y_series


def make_lag_features(series: pd.Series, lags: int = 5, n_steps: int = 10, freq: Optional[str] = None) -> pd.DataFrame:
    """Create lag-based features to allow autoregressive multi-step forecasting.

    series: historical pd.Series (indexed by datetime or integer index)
    lags: number of lagged values to use as features (lag1 ... lagN)
    n_steps: number of future rows (timesteps) to create
    freq: optional pandas offset alias to build a DatetimeIndex for the future

    Returns a DataFrame with n_steps rows and columns ['lag1', ..., 'lag{lags}'].
    The first row corresponds to t+1 and uses the most recent history available.

    Note: This helper only builds the feature DataFrame. To produce multi-step
    forecasts you should iteratively call a fitted model using the predictions
    to roll forward the lag window.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if lags < 1:
        raise ValueError("lags must be >= 1")

    # build last observed lag window
    last_vals = series.values.astype(float)
    if len(last_vals) < lags:
        raise ValueError("series must have at least `lags` observations")

    # prepare index for the future rows
    if isinstance(series.index, pd.DatetimeIndex) and freq is None:
        try:
            freq = pd.infer_freq(series.index)
        except Exception:
            freq = None
    if isinstance(series.index, pd.DatetimeIndex) and freq is not None:
        start = series.index[-1]
        future_index = pd.date_range(start=start, periods=n_steps + 1, freq=freq)[1:]
    else:
        future_index = pd.RangeIndex(start=0, stop=n_steps)

    # create the feature matrix for iterative forecasting
    X_future = []
    window = list(last_vals[-lags:])  # newest last
    for _ in range(n_steps):
        # features are lag1 = value at t, lag2 = t-1, ... so reverse window
        features = window[::-1]
        X_future.append(features)
        # placeholder for next step: we will append predictions externally
        # shift the window (drop oldest, append placeholder)
        window = window[1:] + [np.nan]

    cols = [f"lag{i}" for i in range(1, lags + 1)]
    df = pd.DataFrame(X_future, columns=cols, index=future_index)
    return df


# Convenience function combining fit + iterative forecast using lag features
def project_with_autoregression(
    series: pd.Series,
    lags: int = 5,
    n_steps: int = 10,
    gp_kwargs: Optional[dict] = None,
) -> pd.Series:
    """Fit a GP using lag features formed from `series` and produce an
    n_steps multi-step forecast by iteratively feeding predictions back as lags.

    Returns a pd.Series indexed by inferred future DatetimeIndex (or RangeIndex).
    """
    if gp_kwargs is None:
        gp_kwargs = {}

    # build training dataset from available history
    if len(series) <= lags:
        raise ValueError("series must contain more observations than `lags`")

    # Build supervised dataset: for t in [lags..end-1], X_t = lags previous, y_t = value at t
    X_rows = []
    y_rows = []
    vals = series.values.astype(float)
    for t in range(lags, len(vals)):
        window = vals[t - lags : t]
        X_rows.append(window[::-1])
        y_rows.append(vals[t])

    X = pd.DataFrame(X_rows, columns=[f"lag{i}" for i in range(1, lags + 1)])
    y = pd.Series(y_rows, index=series.index[lags:], name=series.name or "y")

    model = GaussianProcessProjection(**gp_kwargs)
    model.fit(X, y)

    # prepare iterative forecasting
    X_future = make_lag_features(series, lags=lags, n_steps=n_steps)
    preds = []
    window = list(vals[-lags:])
    for i in range(n_steps):
        # build features for this step
        features = pd.DataFrame([window[::-1]], columns=X.columns)
        yhat = model.predict(features).iloc[0]
        preds.append(yhat)
        # slide the window and append prediction
        window = window[1:] + [yhat]

    # build index for result (same as X_future.index)
    result = pd.Series(preds, index=X_future.index, name=series.name or "gp_forecast")
    return result

