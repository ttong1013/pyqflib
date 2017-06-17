# bonds.py
# author: Tony Tong (ttong1013 @github)
# created: 6/10/2017
# last update: 6/16/2017


import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
from scipy import interpolate


def get_rates_from_discounts(T, Z, Tnew=np.array([]), t=0, kind='simple', k=0,
                             order=5, ret_model=False):
    """
    Calculate the spot rates and instantaneous forward rates from discount rate curve Z
    Input:
        T       Input time points of discount rates
        Z       Discount rates (i.e., par = 1)
        Tnew    New time grid points to interpolate the rates.  If Tnew is not provided,
                original T will be used.
        t       Current reference time (default t=0)
        kind    'simple': direct evaluation using discount curve
                'polynomial': fit a polynomial of nth order to log discount curve
                'Nelson-Siegel': fit Nelson-Siegel five-parameter model
                'Svensson': fit Svensson model
        k       Compound frequency per year (k=0 refers to continuous compounding)
        order   Polynomial fit order

    Returns:
        r_spot_new  Spot rates
        r_fwd_new   Instantaneous forward rates

    """

    if kind == 'simple':  # direct evaluation of spot rate from discount curve
        # Evaluate the direct spot rates on the original time grid points from discount curve
        if k == 0:  # continuous compounding
            r_spot = - np.log(Z) / (T - t)  # continuous compounding
        else:  # compounding k times per year
            r_spot = k * (Z ** (-1 / (k * (T - t))) - 1)  # k compounding per year

        # Define interpolation function
        f_x = interpolate.interp1d(T, r_spot, kind='cubic', fill_value='extrapolate')

        # Interpolate spot rates to new time grid points (if provided)
        if len(Tnew) == 0:  # Tnew not specified
            r_spot_new = f_x(T)  # Interpolate to orignal time grid points
        else:
            r_spot_new = f_x(Tnew)  # Interpolate to the new time grid points

        # Create a denser grid points to interpolate the forward rates
        T2 = np.linspace(T.min(), T.max(), 1001)
        f_x2 = interpolate.interp1d(T, r_spot, kind='cubic', fill_value='extrapolate')
        r_spot2 = f_x2(T2)
        d_r_spot2 = np.diff(r_spot2) / np.diff(T2)  # First derivative of the spot rate curve
        r_fwd2 = r_spot2[1:] + d_r_spot2 * (T2[1:] - t)  # Forward rate curve

        # Interpolate back to the new time grid (if provided)
        f_x = interpolate.interp1d(T2[1:], r_fwd2, kind='cubic', fill_value='extrapolate')
        if len(Tnew) == 0:  # Tnew not specified
            r_fwd_new = f_x(T)
        else:
            r_fwd_new = f_x(Tnew)

        return r_spot_new, r_fwd_new


    if kind == 'polynomial':  # use polynomial of nth order to fit r_spot
        y = np.log(Z)
        # Construct the X feature matrix
        X0 = T.reshape([T.shape[0], 1])
        X = np.array([])
        for i in range(order):
            if i == 0:
                X = X0
            else:
                X = np.append(X, X0 ** (i + 1), axis=1)

        model = sm.OLS(y, X).fit(cov_type='HC0')
        coeff = pd.DataFrame(model.params, index=(['a', 'b', 'c', 'd', 'e']), columns=['coeff'])
        print('Polynomial coeffs: \n', coeff)
        if len(Tnew) == 0:  # Tnew not specified
            logZ_pred = model.predict()  # Interpolate to orignal time grid points
            Tnew = T
        else:
            X0_new = Tnew.reshape([Tnew.shape[0], 1])
            X_new = np.array([])
            for i in range(order):
                if i == 0:
                    X_new = X0_new
                else:
                    X_new = np.append(X_new, X0_new ** (i + 1), axis=1)
            logZ_pred = model.predict(X_new)  # Interpolate to the new time grid points

        if k == 0:  # continuous compounding
            r_spot_new = - logZ_pred / (Tnew - t)  # continous compounding
        else:  # compounding k times per year
            r_spot_new = k * (np.exp(logZ_pred) ** (-1 / (k * (Tnew - t))) - 1)

        r_spot_new[0] = r_spot_new[1] - Tnew[1] * (r_spot_new[1] - r_spot_new[2]) / \
                                        (Tnew[1] - Tnew[2])  # extrapolate to remove nan

        # Create a denser grid points to interpolate the forward rates
        T2 = np.linspace(Tnew.min(), Tnew.max(), 1001)
        f_x2 = interpolate.interp1d(Tnew, r_spot_new, kind='cubic', fill_value='extrapolate')
        r_spot2 = f_x2(T2)
        d_r_spot2 = np.diff(r_spot2) / np.diff(T2)  # First derivative of the spot rate curve
        r_fwd2 = r_spot2[1:] + d_r_spot2 * (T2[1:] - t)  # Forward rate curve

        # Interpolate back to the new time grid (if provided)
        f_x = interpolate.interp1d(T2[1:], r_fwd2, kind='cubic', fill_value='extrapolate')
        if len(Tnew) == 0:  # Tnew not specified
            r_fwd_new = f_x(T)
        else:
            r_fwd_new = f_x(Tnew)

        if ret_model:
            return r_spot_new, r_fwd_new, model
        else:
            return r_spot_new, r_fwd_new


    def func_NS_err(x, T, r_spot):
        # Unwrap the parameters
        beta0 = x[0]
        beta1 = x[1]
        beta2 = x[2]
        tau1 = x[3]
        tau2 = x[4]
        T = np.array(T)
        r_fit = beta0 + beta1 * ((1 - np.exp(-T / tau1)) / (T / tau1)) \
                + beta2 * (((1 - np.exp(-T / tau2)) / (T / tau2)) - np.exp(-T / tau2))
        weight = 1 / T
        err = ((r_fit - r_spot) ** 2 * weight).sum()
        return err

    if kind == 'Nelson-Siegel':
        # first derive the direct spot rate from Z curve, then we perform the fit
        if k == 0:  # continuous compounding
            r_spot = - np.log(Z) / (T - t)  # continous compounding
        else:  # compounding k times per year
            r_spot = k * (Z ** (-1 / (k * (T - t))) - 1)

        x0 = [0.048, -0.012, 0.02, 4.8, 0.49]
        # It is critical to set 'ftol' to 1e-14
        res = sp.optimize.minimize(func_NS_err, x0, args=(T, r_spot), method='SLSQP', options={'ftol': 1e-14})
        print(res)
        x = res.x
        beta0 = x[0]
        beta1 = x[1]
        beta2 = x[2]
        tau1 = x[3]
        tau2 = x[4]

        if len(Tnew) == 0:  # Tnew not specified
            Tnew = T

        r_spot_fit = beta0 + beta1 * ((1 - np.exp(-Tnew / tau1)) / (Tnew / tau1)) \
                     + beta2 * (((1 - np.exp(-Tnew / tau2)) / (Tnew / tau2)) - np.exp(-Tnew / tau2))

        r_fwd_fit = beta0 + beta1 * np.exp(-Tnew / tau1) + beta2 * (Tnew / tau2) * np.exp(-Tnew / tau2)

        if np.isnan(r_spot_fit[0]):
            r_spot_fit[0] = r_spot[0]

        if ret_model:
            return r_spot_fit, r_fwd_fit, res
        else:
            return r_spot_fit, r_fwd_fit

    def func_S_err(x, T, r_spot):
        # Unwrap the parameters
        beta0 = x[0]
        beta1 = x[1]
        beta2 = x[2]
        beta3 = x[3]
        tau1 = x[4]
        tau2 = x[5]
        T = np.array(T)
        r_fit = beta0 + beta1 * ((1 - np.exp(-T / tau1)) / (T / tau1)) \
                + beta2 * (((1 - np.exp(-T / tau1)) / (T / tau1)) - np.exp(-T / tau1)) \
                + beta3 * (((1 - np.exp(-T / tau2)) / (T / tau2)) - np.exp(-T / tau2))
        weight = 1 / T
        err = np.array(((r_fit - r_spot) ** 2 * weight)).sum()
        return err

    if kind == 'Svensson':
        # first derive the direct spot rate from Z curve, then we perform the fit
        if k == 0:  # continuous compounding
            r_spot = - np.log(Z) / (T - t)  # continuous compounding
        else:  # compounding k times per year
            r_spot = k * (Z ** (-1 / (k * (T - t))) - 1)

        x0 = [0.048, -0.012, 0.02, 0.02, 4.8, 0.9]
        # It is critical to set 'ftol' to 1e-14
        res = sp.optimize.minimize(func_S_err, x0, args=(T, r_spot), method='SLSQP', options={'ftol': 1e-15})
        print(res)
        x = res.x
        beta0 = x[0]
        beta1 = x[1]
        beta2 = x[2]
        beta3 = x[3]
        tau1 = x[4]
        tau2 = x[5]

        if len(Tnew) == 0:  # Tnew not specified
            Tnew = T

        r_spot_fit = beta0 + beta1 * ((1 - np.exp(-Tnew / tau1)) / (Tnew / tau1)) \
                     + beta2 * (((1 - np.exp(-Tnew / tau1)) / (Tnew / tau1)) - np.exp(-Tnew / tau1)) \
                     + beta3 * (((1 - np.exp(-Tnew / tau2)) / (Tnew / tau2)) - np.exp(-Tnew / tau2))

        r_fwd_fit = beta0 + beta1 * np.exp(-Tnew / tau1) + beta2 * (Tnew / tau1) * np.exp(-Tnew / tau1) \
                    + beta3 * (Tnew / tau2) * np.exp(-Tnew / tau2)

        if np.isnan(r_spot_fit[0]):
            r_spot_fit[0] = r_spot[0]
        # print(r_spot_fit[0], r_spot[0])

        if ret_model:
            return r_spot_fit, r_fwd_fit, res
        else:
            return r_spot_fit, r_fwd_fit



def get_period_rates(T, r_inst, Tnew=np.array([]), period=0.25, t=0):
    """
    Calculate the period rates from instant rates numerical integration
    For example:
    forward period rates f(t, T1, T2) from instantaneous forward rates f(t, T1)
    Input:
        T
    """
    from scipy import interpolate

    # Define an interpolation function for the instantaneous rates
    f_x = interpolate.interp1d(T, r_inst, kind='cubic', fill_value='extrapolate')

    if len(Tnew) == 0:
        Tnew = T
    r_period = np.zeros(Tnew.shape)
    for i in range(len(Tnew)):
        nn = 21
        T2 = np.linspace(T[i], T[i] + period, nn)
        r_inst2 = f_x(T2)
        r_period[i] = r_inst2.mean()

    return r_period



def get_par_yield(T, r_spot, Tnew=[], k=2, t=0):
    """Calculate the par yield curve"""

    Tcopy = T.copy()
    if Tcopy[0] == 0:
        Tcopy[0] = 0.001

    f_x2 = interpolate.interp1d(Tcopy, r_spot, kind='cubic', fill_value='extrapolate')
    # Interpolate to a denser intermediate grid
    T2 = np.linspace(Tcopy.min(), Tcopy.max(), 1001)
    r_spot2 = f_x2(T2)

    nc = np.floor((T2 - t) * 2)
    r_par2 = np.zeros(r_spot2.shape)
    for i in range(T2.shape[0]):
        t_coupon_i = np.r_[T2[i]:0:-(1. / k)][::-1]
        #         print('Coupon time', t_coupon_i)
        r_spot_i = f_x2(t_coupon_i)
        #         print('Spot rate at coupon time', r_spot_i)
        if k == 0:
            disc_i = np.exp(-(T2[i] - t_coupon_i) * r_spot_i)  # Use continuous compounding for spot rate
        else:
            disc_i = (1 + r_spot_i / k) ** (-k * (T2[i] - t_coupon_i))  # Use semi-annual compounding for spot rate
        # print('Discount at coupon time', disc_i)
        if k == 0:
            r_par2[i] = 2 * (
                1 - np.exp(-(T2[i] - t) * r_spot2[i])) / disc_i.sum()  # Use continuous compounding for spot rate
        else:
            r_par2[i] = 2 * (
                1 - (1 + r_spot2[i] / k) ** (
                    -k * (T2[i] - t))) / disc_i.sum()  # Use semi-annual compounding for spot rate
            #         print('i = ', i)
            #         print(disc_i)
            #         print((1 - (1 + r_spot2[i]/k)**(-k*(T2[i]-t))))

    # Need to compensate for accured interest
    r_par2 = (np.ceil(T2 / 0.5) / 2) / T2 * r_par2

    # Interpolate back to the original time grid (or otherwise specified)
    f_x = interpolate.interp1d(T2, r_par2, kind='cubic', fill_value='extrapolate')
    if len(Tnew) == 0:  # Tnew not specified
        r_par = f_x(T)
    else:
        r_par = f_x(Tnew)

    return r_par


def main():
    """Perform test functions"""
    import matplotlib.pyplot as plt

    data0 = pd.read_excel('bonds_discount_data.xls', skiprows=0)
    T = data0['Maturity'].values
    price = data0['Price'].values
    Z = price / 100  # par=100

    # New time grid points of 3-mo intervals
    T7 = np.r_[0:T.max():0.25]

    # Calculate spot rate and instantaneous forward rates
    r_spot, r_fwd = get_rates_from_discounts(T, Z, k=2)
    r_spot_7, r_fwd_7 = get_rates_from_discounts(T, Z, Tnew=T7, k=2)

    r_fwd_7_3mo = get_period_rates(T7, r_fwd_7, period=0.25, t=0)
    r_par_7 = get_par_yield(T7, r_spot_7, T7)
    r_par_7_5yr = get_period_rates(T7, r_par_7, period=5., t=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(T7, r_spot_7, color='b', linewidth=3, label=r'Spot Rate: $r_2(0,T)$', alpha=0.6)
    # ax.plot(T, r_spot_sm, color='g', linewidth=3, label='Spot Rate (smoothed)')
    ax.plot(T7, r_fwd_7, color='r', linestyle='-', linewidth=2, label=r'Forward Rate (inst): $f_2(0, T)$', alpha=0.6)
    ax.plot(T7, r_fwd_7_3mo, color='darkred', linestyle='--', linewidth=2.5,
            label=r'Forward Rate (3-mo): $f_2(0, T, T+0.25)$',
            alpha=0.6)
    ax.plot(T7, r_par_7, color='m', linewidth=2, label='Par Yield', alpha=0.6)
    ax.plot(T7[:80], r_par_7_5yr[:80], color='c', linestyle='--', linewidth=2.5, label='Par Yield (5yr)', alpha=0.6)
    plt.xlabel('Maturity (years)')
    plt.ylabel('Interest Rate')
    plt.ylim([0.01, 0.08])
    plt.legend(loc=0)

    plt.draw()
    plt.show()


if __name__ == '__main__':
    main()
