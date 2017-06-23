# bonds.py
# author: Tony Tong (ttong1013 @github)
# created: 6/15/2017
# last update: 6/23/2017
import numpy as np
from scipy.stats import norm


def BSM(S, K, T, sigma, r, y=0, t=0, kind='c', ret='option'):
    """
    Implement Black-Scholes-Merton formula for plain-vanilla European Call or Put options
    Function can also take vectorized input, however, their dimensions need to match
    kind: 'c' for Calls and 'p' for Puts
    return: 'option' returns only the option value
            'complete' returns the option value and other binary option values
    """
    assert kind.lower() == 'c' or kind.lower() == 'p', \
        "Option type is wrong, only 'C' or 'P' is allowed"
    assert sigma > 0 and S > 0 and K > 0 and T > 0

    N = norm.cdf
    sigma_rt = sigma * np.sqrt(T - t)
    x1 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt + sigma_rt / 2  # x+
    x2 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt - sigma_rt / 2  # x-

    # Binary options values
    AONC = S * np.exp(-y * (T - t)) * N(x1)  # Asset-Or-Nothing-Call
    AONP = S * np.exp(-y * (T - t)) * N(-x1)  # Asset-Or-Nothing-Put
    CONC = np.exp(-r * (T - t)) * N(x2)  # Cash-Or-Nothing-Call
    CONP = np.exp(-r * (T - t)) * N(-x2)  # Cash-Or-Nothing-Put

    if kind.lower() == 'c':
        val = AONC - K * CONC
    elif kind.lower() == 'p':
        val = K * CONP - AONP
    else:
        val = np.nan

    if ret == 'complete':
        return val, CONC, AONC, CONP, AONP
    else:
        return val


def BSM_Delta(S, K, T, sigma, r, y=0, t=0, kind='c', ret='option'):
    """
    Evaluate the Delta with respect to S and K using analytical expressions
    """
    assert kind.lower() == 'c' or kind.lower() == 'p', \
        "Option type is wrong, only 'C' or 'P' is allowed"
    assert sigma > 0 and S > 0 and K > 0 and T > 0

    N = norm.cdf
    sigma_rt = sigma * np.sqrt(T - t)
    x1 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt + sigma_rt / 2  # x+
    x2 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt - sigma_rt / 2  # x-

    if kind.lower() == 'c':
        Delta_wrt_S = np.exp(-y * (T-t)) * N(x1)
        Delta_wrt_K = -np.exp(-r * (T-t)) * N(x2)
    elif kind.lower() == 'p':
        Delta_wrt_S = -np.exp(-y * (T-t)) * N(-x1)
        Delta_wrt_K = np.exp(-r * (T-t)) * N(-x2)
    else:
        Delta_wrt_S = np.nan
        Delta_wrt_K = np.nan
    return Delta_wrt_S, Delta_wrt_K

