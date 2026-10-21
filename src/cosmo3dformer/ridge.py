"""Frozen-representation evaluation with ridge regression.

The protocol is fixed: features are standardised on the training split only,
the target transform is fitted on training labels only, the penalty is chosen
on validation in the original property units, and the model is then refitted on
the training split with the chosen penalty. Validation is used for the penalty
and for nothing else.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .splits import Target

ALPHAS = (0.1, 1.0, 10.0, 100.0)


def r2_score(y_true, y_pred):
    """Coefficient of determination as 1 - SSE/SST."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


def rmse(y_true, y_pred):
    """Root mean squared error in the units of y_true."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def fit_predict(x_train, y_train, x_val, y_val, x_test, transform="standard",
                alphas=ALPHAS):
    """Run the frozen-representation protocol for one fold.

    Returns the test predictions in original units, the chosen penalty and the
    validation score at that penalty. Ties in the validation score keep the
    smaller penalty because the comparison is strictly greater than.
    """
    scaler = StandardScaler().fit(x_train)
    xt = scaler.transform(x_train)
    xv = scaler.transform(x_val)
    xs = scaler.transform(x_test)

    target = Target(transform).fit(y_train)
    zt = target.forward(y_train)

    best_alpha, best_score = None, -np.inf
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(xt, zt)
        score = r2_score(y_val, target.inverse(model.predict(xv)))
        if score > best_score:
            best_score, best_alpha = score, alpha

    model = Ridge(alpha=best_alpha).fit(xt, zt)
    return target.inverse(model.predict(xs)), best_alpha, float(best_score)
