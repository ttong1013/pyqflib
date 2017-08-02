# options.py
# author: Tony Tong (ttong1013 @github)
# created: 6/15/2017
# last update: 6/29/2017
#
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

@jit
def BSM(S, K, T, sigma, r, y=0., t=0., kind='c', ret='option'):
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
    Function can also take vectorized input, however, their dimensions need to match
    """
    assert kind.lower() == 'c' or kind.lower() == 'p', \
        "Option type is wrong, only 'C' or 'P' is allowed"
    assert sigma > 0 and S > 0 and K > 0 and T > 0

    N = norm.cdf
    sigma_rt = sigma * np.sqrt(T - t)
    x1 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt + sigma_rt / 2  # x+
    x2 = (np.log(S / K) + (r - y) * (T - t)) / sigma_rt - sigma_rt / 2  # x-

    if kind.lower() == 'c':
        Delta_wrt_S = np.exp(-y * (T - t)) * N(x1)
        Delta_wrt_K = -np.exp(-r * (T - t)) * N(x2)
    elif kind.lower() == 'p':
        Delta_wrt_S = -np.exp(-y * (T - t)) * N(-x1)
        Delta_wrt_K = np.exp(-r * (T - t)) * N(-x2)
    else:
        Delta_wrt_S = np.nan
        Delta_wrt_K = np.nan
    return Delta_wrt_S, Delta_wrt_K


@jit
def draw_uniform(size, low=0., high=1., seed=42):
    """
    Draw uniformly distributed random numbers
    It is critical a new seed is provided to draw different results
    """
    np.random.seed(seed)
    return np.random.uniform(low, high, size)


@jit
def draw_normal(size, mu=0., sigma=1., seed=42):
    """
    Draw normally distributed random numbers
    It is critical a new seed is provided to draw different results
    """
    np.random.seed(seed)
    return np.random.normal(loc=mu, scale=sigma, size=size)


def optionMC_one_step(S, K, T, sigma, r, y=0., t=0., kind='c', alpha_s=1.0, alpha_k=1.0, alpha_tot=1.0,
                      n=2**20, m=2**10, subbin=False, antithetic=True, controlvariate=False,
                      moment_matching=True, matching_style='log', seed=402):
    """
    Implementation of general Monte-Carlo option pricing tool with one-step configuration (path independent)
    Method does NOT support vectorized operation.
    kind:               'c' or 'C' for European call options
                        'p' or 'P' for European put options
                        'logcall' for LogCall option K[ln(ST/K)]^+
    alpha_s:            (Default 1.0)control the power option features. Power over S_T
    alphs_k:            (Default 1.0)control the power option features. Power over K
    alpha_tot:          (Default 1.0)control the power option features. Overall power.
                        General power option payoff can be expressed as:
                        C_T = max[(S**alpha_s - K**alpha_k)**alpha_tot, 0]
                        P_T = max[(K**alpha_k - S**alpha_s)**alpha_tot, 0]
                        European options have alpha_s=1.0, alpha_k=1.0, alpha_tot=1.0
    antithetic:         (Default True) turn anti-thetic variate option on (True) or off (False)
    moment_matching:    (Default True) turn moment_matching option on (True) or off (False)
    matching_style:     'log' (Default) match the 1st and 2nd log moments
                        'exp' match the 1st and 2nd exponential (true) moments
    n:                  number of discrete sampling
    m:                  size of sub-bins
    subbin:             (Default False)controls whether subbinning to be performed or not

    """
    assert S >= 0 and K >= 0 and T >= 0 and sigma >= 0

    mu = r - y  # drift
    mu_tilde = mu - 0.5 * sigma ** 2  # log drift
    discount_factor = np.exp(-r * (T - t))
    one_over_sqrt_n = 1.0 / np.sqrt(n)
    sigma_sqrt_T = sigma * np.sqrt(T - t)

    x = draw_normal((n, 1), 0, 1, seed)

    if moment_matching:  # After the 1st and 2nd moment matching, it is guaranteed E[x]=0 and Var[x]=1
        # print("Before moment mathcing: ", x.mean(), x.std())
        if matching_style == 'log':
            x = x - x.mean()  # De-mean to match the 1st moment (log mean)
            x = x / x.std()  # Scale/normalize by std dev

        elif matching_style == 'exp':
            b = 1.0

            def f(b):  # define a nested function to find b such that f=0
                return (np.exp(2 * b * sigma_sqrt_T * x)).mean() / (
                                                                       (np.exp(
                                                                           b * sigma_sqrt_T * x)).mean()) ** 2 - np.exp(
                    sigma_sqrt_T ** 2)

            def derivative(f, x, accuracy):
                deriv = (1.0 / (2 * accuracy)) * (f(x + accuracy) - f(x - accuracy))
                return deriv

            def solve(f, approximation, h):
                current_approx = approximation
                prev_approx = float("inf")  # So that the first time the while loop will run
                while np.abs(current_approx - prev_approx) >= h:
                    prev_approx = current_approx
                    current_approx = current_approx - (f(current_approx)) / derivative(f, current_approx, h)
                return current_approx

            b = solve(f, 1.0, 1e-7)
            a = sigma_sqrt_T / 2 - np.log((np.exp(b * sigma_sqrt_T * x)).mean()) / sigma_sqrt_T
            #             print("\n--- Moment matching (exp) params ---")
            #             print("n = {0:d}  \ta = {1:.5e} \tb = {2:5e} \tf(b) = {3:5e}".format(n, a, b, f(b)))
            x = a + b * x


        else:  # matching_style input is wrong
            raise ValueError
            # print("After moment mathcing: ", x.mean(), x.std())

    if antithetic:
        # Evaluate the ST and its antivariate
        ST_1 = S * np.exp(mu_tilde * (T - t) + sigma_sqrt_T * x)
        ST_2 = S * np.exp(mu_tilde * (T - t) - sigma_sqrt_T * x)
        ST = (ST_1 + ST_2) / 2
    else:
        ST = S * np.exp(mu_tilde * (T - t) + sigma_sqrt_T * x)
        ST_1 = ST_2 = ST
    ST_mean = ST.mean()  # Sample mean of ST
    ST_std = ST.std()  # Sample std dev of ST
    ST_mean_std = ST_std * one_over_sqrt_n  # CLT estimator of std dev of mean of ST

    if np.round(alpha_s) == 1.0 and np.round(alpha_k) == 1.0 and np.round(alpha_tot) == 1.0:
        flag_power_option = False
    else:
        flag_power_option = True

    if kind.lower() == 'c':
        if not flag_power_option:  # regular option
            if antithetic:
                CT_1 = np.amax(np.c_[ST_1 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
                CT_2 = np.amax(np.c_[ST_2 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[ST - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
        else:  # general power option
            if antithetic:
                CT_1 = np.amax(np.c_[(ST_1 ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Call payoff
                CT_2 = np.amax(np.c_[(ST_2 ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Call payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[(ST ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                             axis=1)  # Standard European Call payoff

    elif kind.lower() == 'p':
        if not flag_power_option:  # regular option
            if antithetic:
                CT_1 = np.amax(np.c_[K - ST_1, np.zeros(ST.shape)], axis=1)  # Standard European Put payoff
                CT_2 = np.amax(np.c_[K - ST_2, np.zeros(ST.shape)], axis=1)  # Standard European Put payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[K - ST, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
        else:  # general power option
            if antithetic:
                CT_1 = np.amax(np.c_[(K ** alpha_k - ST_1 ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Put payoff
                CT_2 = np.amax(np.c_[(K ** alpha_k - ST_2 ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Put payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[(K ** alpha_k - ST ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                             axis=1)  # Standard European Call payoff

    elif kind.lower() == 'logcall':
        if antithetic:
            CT_1 = np.amax(np.c_[K * np.log(ST_1 / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff
            CT_2 = np.amax(np.c_[K * np.log(ST_2 / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff
            CT = (CT_1 + CT_2) / 2
        else:  # antithetic==False
            CT = np.amax(np.c_[K * np.log(ST / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff

        if controlvariate:
            if antithetic:
                CT_1_vc = np.amax(np.c_[ST_1 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
                CT_2_vc = np.amax(np.c_[ST_2 - K, np.zeros(ST.shape)],
                                  axis=1)  # Standard European Call payoff, antithetic
                CT_vc = (CT_1_vc + CT_2_vc) / 2
            else:  # antithetic==False
                CT_vc = np.amax(np.c_[ST - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
            CT = CT - CT_vc + BSM(S=S, K=K, T=T, sigma=sigma, r=r, y=y, t=t, kind='c',
                                  ret='option') / discount_factor

    ## C represents general derivitive with a payoff function
    # Time T payoff stats
    CT_mean = CT.mean()
    CT_std = CT.std()
    CT_mean_std = CT_std * one_over_sqrt_n  # by CLT estimate

    # Time t payoff stats
    Ct_mean = CT_mean * discount_factor
    Ct_mean_std = CT_mean_std * discount_factor  # by CLT estimate

    if subbin and n > m:
        k = int(n / m)
        ST_mean_std_star = ST[0:k * m].reshape((-1, m)).mean(axis=1).std() / np.sqrt(k)
        CT_mean_std_star = CT[0:k * m].reshape((-1, m)).mean(axis=1).std() / np.sqrt(k)
        Ct_mean_std_star = CT_mean_std_star * discount_factor
    else:
        Ct_mean_std_star = CT_mean_std_star = ST_mean_std_star = np.nan

    if subbin:
        return Ct_mean, Ct_mean_std, Ct_mean_std_star, CT_mean, CT_mean_std, \
               CT_mean_std_star, ST_mean, ST_mean_std, ST_mean_std_star
    else:
        return Ct_mean, Ct_mean_std, CT_mean, CT_mean_std, ST_mean, ST_mean_std


def MC_eval(func, S, K, T, sigma, r, y=0, t=0, kind='c', n_start=5, n_end=20, subbin=True,
            antithetic=True, moment_matching=True, matching_style='log', rep=1, seed=42):
    from time import time

    np.random.seed(seed)
    n_list = [2 ** i for i in range(n_start, n_end + 1)]
    output = []
    for n in n_list:
        output_i = []
        for kk in range(rep):   # Repeat the core function 'rep' times to evaluate the realized std err
            t0 = time()
            output_kk = func(
                S, K, T, sigma, r, y, kind=kind, n=n, subbin=subbin, antithetic=antithetic,
                moment_matching=moment_matching, matching_style=matching_style, seed=np.random.randint(100000)
            )
            t1 = time()
            exec_time = t1 - t0
            epsilon = 1e5 * exec_time * output_kk[1] ** 2
            output_kk = list(output_kk)
            output_kk.extend([exec_time, epsilon])
            output_i.append(output_kk)

        output_i_mean = np.array(output_i).mean(axis=0)
        output_i_std = np.array(output_i).std(axis=0)
        output.append(list(np.r_[output_i_mean, output_i_std]))

    n_df = pd.DataFrame(n_list, columns=['n'])
    output_df = pd.DataFrame(output, columns=['Ct_mean', 'Ct_mean_std', 'Ct_mean_std*',
                                              'CT_mean', 'CT_mean_std', 'CT_mean_std*',
                                              'ST_mean', 'ST_mean_std', 'ST_mean_std*',
                                              'time', 'epsilon',
                                              'Ct_mean_sd', 'Ct_mean_std_sd', 'Ct_mean_std*_sd',
                                              'CT_mean_sd', 'CT_mean_std_sd', 'CT_mean_std*_sd',
                                              'ST_mean_sd', 'ST_mean_std_sd', 'ST_mean_std*_sd',
                                              'time_std', 'epsilon_std'])
    total_df = pd.concat([n_df, output_df], axis=1)
    total_df.sort_values(by='n', inplace=True)
    return total_df


def test_wrapper2(func=optionMC_one_step, S=100., K=110., T=1., sigma=0.16,
                  r=0.1, y=0.0, kind='c', t=0, n_start=5, n_end=20, subbin=True,
                  antithetic=True, moment_matching=True, matching_style='log',
                  rep=10, seed=42, title=''):
    # Calculate the exact values of ST and Ct
    ST_true = S * np.exp((r - y) * (T-t))
    Ct_true = BSM(S=S, K=K, T=T, sigma=sigma, r=r, y=y, kind=kind)

    total_df = MC_eval(func, S=S, K=K, T=T, sigma=sigma,
                       r=r, y=y, kind=kind, n_start=n_start, n_end=n_end, subbin=True, antithetic=antithetic,
                       moment_matching=moment_matching, matching_style=matching_style, rep=rep, seed=seed)

    lg_n = np.log10(total_df['n'].values)
    lg_ST_abs_err = np.log10(np.abs(total_df['ST_mean'].values - ST_true))
    lg_ST_mean_std = np.log10(total_df['ST_mean_std'].values)
    lm = sm.OLS(lg_ST_mean_std, sm.add_constant(lg_n)).fit()
    lg_ST_mean_std_fit = lm.predict()
    slope_ST_mean_std = lm.params[1]
    lg_ST_mean_std_star = np.log10(total_df['ST_mean_sd'].values)

    lg_Ct_abs_err = np.log10(np.abs(total_df['Ct_mean'].values - Ct_true))
    lg_Ct_mean_std = np.log10(total_df['Ct_mean_std'].values)
    lm = sm.OLS(lg_Ct_mean_std, sm.add_constant(lg_n)).fit()
    lg_Ct_mean_std_fit = lm.predict()
    slope_Ct_mean_std = lm.params[1]
    lg_Ct_mean_std_star = np.log10(total_df['Ct_mean_sd'].values)

    time = total_df['time'].values

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    #     ax.plot(lg_n, lg_ST_abs_err, 'bo', label=r'MC $\left|<S_T>_n  - E[S_T]\right|$')
    ax.plot(lg_n, lg_ST_mean_std, 'bo', linestyle='-', markerfacecolor='None', markersize=10,
            label=r'MC $std<S_T>_n$ est by CLT', alpha=0.6)
    ax.plot(lg_n, lg_ST_mean_std_star, color='b', linestyle='--', marker='s', markersize=12,
            markerfacecolor='None', markeredgecolor='r', label=r'MC $std<S_T>_n$ realized', alpha=0.7)

    #     ax.plot(lg_n, lg_Ct_abs_err, 'go', label=r'MC $\left|<C_t>_n  - E[C_t]\right|$')
    ax.plot(lg_n, lg_Ct_mean_std, 'go', linestyle='-', markerfacecolor='None', markersize=10,
            label=r'MC $std<C_t>_n$ est by CLT', alpha=0.6)
    ax.plot(lg_n, lg_Ct_mean_std_star, color='g', linestyle='--', marker='s', markersize=12,
            markerfacecolor='None', markeredgecolor='c', label=r'MC $std<S_T>_n$ realized', alpha=0.7)
    ax.set_xlim([1., 6.5])
    ymin, ymax = ax.get_ylim()
    if ymin < -6:
        ymin = -6
        ax.set_ylim([ymin, ymax])
    ax.text(1.1, ymin + 0.6, 'slope[ST_mean_std] = {0:.2f}'.format(slope_ST_mean_std), fontsize=15)
    ax.text(1.1, ymin + 0.2, 'slope[Ct_mean_std] = {0:.2f}'.format(slope_Ct_mean_std), fontsize=15)
    plt.xlabel(r'$log_{10}(n)$')
    plt.ylabel(r'$log_{10}(Err)$')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.draw()
    return total_df


def grouped_test2():
    # CLT Est Std Err vs Realized Std Err
    rep = 20
    summary = test_wrapper2(func=optionMC_one_step, S=100., K=100., T=1., sigma=0.2,
                            r=0.01, y=0.02, kind='c', n_start=4, n_end=20, subbin=True,
                            antithetic=False, moment_matching=False, rep=rep, seed=400,
                            title='BBMC (MM off; AV off)')

    summary = test_wrapper2(func=optionMC_one_step, S=100., K=100., T=1., sigma=0.2,
                            r=0.01, y=0.02, kind='c', n_start=4, n_end=20, subbin=True,
                            antithetic=True, moment_matching=False, rep=rep, seed=400,
                            title='AMC(MM off; AV on)')

    summary = test_wrapper2(func=optionMC_one_step, S=100., K=100., T=1., sigma=0.2,
                            r=0.01, y=0.02, kind='c', n_start=4, n_end=20, subbin=True,
                            antithetic=False, moment_matching=True, matching_style='log',
                            rep=rep, seed=400, title='MMMC (MM (log) on; AV off)')

    summary = test_wrapper2(func=optionMC_one_step, S=100., K=100., T=1., sigma=0.2,
                            r=0.01, y=0.02, kind='c', n_start=4, n_end=20, subbin=True,
                            antithetic=False, moment_matching=True, matching_style='exp',
                            rep=rep, seed=400, title='MMMC (MM (exp) on; AV off)')

    summary = test_wrapper2(func=optionMC_one_step, S=100., K=100., T=1., sigma=0.2,
                            r=0.01, y=0.02, kind='c', n_start=4, n_end=20, subbin=True,
                            antithetic=True, moment_matching=True, matching_style='log',
                            rep=rep, seed=400, title='AMMMC (MM (log) on; AV on)')


def optionMC_multi_step(S, K, T, sigma, r, y=0., t=0., kind='c', Ta=0., alpha_s=1.0, alpha_k=1.0, alpha_tot=1.0,
                        n=2 ** 20, antithetic=True, controlvariate=False, control='gc', control_loading=False,
                        moment_matching=True, matching_style='log', seed=402):
    """
    Implementation of general Monte-Carlo option pricing tool with multi-step configuration (path-dependent)
    under Black-Scholes framework (constant parameters)
    kind:       'c' or 'C' for European call options and 'p' or 'P' for European put options
                'logcall' for LogCall option K[ln(ST/K)]^+
    alpha_s:    control the power option features, by default equal to 1.0
    alphs_k:    control the power option features, by default equal to 1.0
    alpha_tot:  control the power option features, by default equal to 1.0
                General power option payoff:
                C_T = max[(S**alpha_s - K**alpha_k)**alpha_tot, 0]
                P_T = max[(K**alpha_k - S**alpha_s)**alpha_tot, 0]
    antithetic: turn anti-thetic variate option on (True) or off (False)
    moment_matching:  turn moment_matching option on (True) or off (False)
    matching_style:   'log' match the 1st and 2nd log moments
                      'exp' match the 1st and 2nd exponential (true) moments
    controlvariate:   control variate option
    n:          number of discrete sampling paths
    T:          a numpy 1-D array of time-stepping points

    """
    assert S >= 0 and K >= 0 and sigma >= 0
    assert len(T) >= 1  # T should contain at least one time point

    mu = r - y  # drift, scalar
    mu_tilde = mu - 0.5 * sigma ** 2  # log drift, scalar

    if T[0] == t:
        T2 = T.copy()
    else:
        T2 = np.r_[t, T]

    delta_t = np.diff(T2)  # vector of length m
    m = len(delta_t)  # total number of time steps, note the step size may vary

    discount_factor = np.exp(-r * delta_t)  # incremental discount factor, vector of length m
    one_over_sqrt_n = 1.0 / np.sqrt(n)
    sigma_sqrt_delta_t = sigma * np.sqrt(delta_t)  # incremental vol, vector of length m

    x = draw_normal((n, m), 0, 1, seed)  # standard normal random shocks (nxm)

    if moment_matching:  # After the 1st and 2nd moment matching, it is guaranteed E[x]=0 and Var[x]=1
        # print("Before moment mathcing: ", x.mean(axis=1), x.std(=axis=1))
        if matching_style == 'log':
            x = x - x.mean(axis=0)  # De-mean to match the 1st moment (log mean)
            x = x / x.std(axis=0)  # Scale/normalize by std dev

        elif matching_style == 'exp':
            pass

        else:  # matching_style input is wrong
            raise ValueError
            # print("After moment mathcing: ", x.mean(axis=1), x.std(axis=1))

    if antithetic:
        # Evaluate the ST and its antivariate
        deltaSt_1 = np.exp(mu_tilde * delta_t + sigma_sqrt_delta_t * x)  # nxm
        St_1 = S * deltaSt_1.cumprod(axis=1)

        deltaSt_2 = np.exp(mu_tilde * delta_t - sigma_sqrt_delta_t * x)  # nxm
        St_2 = S * deltaSt_2.cumprod(axis=1)

        St = (St_1 + St_2) / 2  # nxm

        # Terminal values
        ST_1 = St_1[:,-1]
        ST_2 = St_2[:, -1]
        ST = St[:, -1]  # Terminal stock price

    else:
        deltaSt = np.exp(mu_tilde * delta_t + sigma_sqrt_delta_t * x)  # nxm
        St = S * deltaSt.cumprod(axis=1)  # nxm
        ST = St[:, -1]  # Terminal stock price

    St_mean = St.mean(axis=0)  # Sample mean of ST, (m,)
    ST_mean = St_mean[-1]
    St_std = St.std(axis=0)  # Sample std dev of ST, (m,)
    ST_std = St_std[-1]
    St_mean_std = St_std * one_over_sqrt_n  # CLT estimator of std dev of mean of ST, (m,)
    ST_mean_std = St_mean_std[-1]

    if np.round(alpha_s) == 1.0 and np.round(alpha_k) == 1.0 and np.round(alpha_tot) == 1.0:
        flag_power_option = False
    else:
        flag_power_option = True

    if kind.lower() == 'c':
        if flag_power_option == False:  # regular option
            if antithetic:
                CT_1 = np.amax(np.c_[ST_1 - K, np.zeros(ST_1.shape)], axis=1)  # Standard European Call payoff
                CT_2 = np.amax(np.c_[ST_2 - K, np.zeros(ST_2.shape)], axis=1)  # Standard European Call payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[ST - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
        else:  # general power option
            if antithetic:
                CT_1 = np.amax(np.c_[(ST_1 ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Call payoff
                CT_2 = np.amax(np.c_[(ST_2 ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Call payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[(ST ** alpha_s - K ** alpha_k) ** alpha_tot, np.zeros(ST.shape)],
                             axis=1)  # Standard European Call payoff

    elif kind.lower() == 'ac':  # Asian arithmic call
        asian_idx = (T2[1:] > T2[-1] - Ta)  # remove the 0 at the beginning
        ma = asian_idx.sum()
        if antithetic:
            STa_1 = St_1[:,asian_idx].mean(axis=1)
            STa_2 = St_2[:,asian_idx].mean(axis=1)

            CT_1 = np.amax(np.c_[STa_1 - K, np.zeros(STa_1.shape)], axis=1)  # Standard European Call payoff
            CT_2 = np.amax(np.c_[STa_2 - K, np.zeros(STa_2.shape)], axis=1)  # Standard European Call payoff, antithetic
            CT = (CT_1 + CT_2) / 2

        else:
            STa = St[:, asian_idx].mean(axis=1)
            CT = np.amax(np.c_[STa - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff

        # Calculate the control variates
        if controlvariate == True:
            if antithetic:
                # vc - vanilla call
                CT_1_vc = np.amax(np.c_[ST_1 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
                CT_2_vc = np.amax(np.c_[ST_2 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff, antithetic
                CT_vc = (CT_1_vc + CT_2_vc) / 2

                # gc - geometric all
                ST_gc_1 = np.exp(np.log(St_1[:,asian_idx]).mean(axis=1))  # geometric mean
                ST_gc_2 = np.exp(np.log(St_2[:,asian_idx]).mean(axis=1))  # geometric mean
                CT_gc_1 = np.amax(np.c_[ST_gc_1 - K, np.zeros(ST_gc_1.shape)], axis=1)  # Standard European Call payoff
                CT_gc_2 = np.amax(np.c_[ST_gc_2 - K, np.zeros(ST_gc_2.shape)], axis=1)  # Standard European Call payoff, antithetic
                CT_gc = (CT_gc_1 + CT_gc_2) / 2

            else:  # antithetic==False
                # vc - vanilla call
                CT_vc = np.amax(np.c_[ST - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff

                # gc - geometric all
                ST_gc = np.exp(np.log(St[:,asian_idx]).mean(axis=1))  # geometric mean
                CT_gc = np.amax(np.c_[ST_gc - K, np.zeros(ST_gc.shape)], axis=1)  # Standard European Call payoff

            # Call using vanilla call as control variate
            if control_loading:
                beta = np.cov(CT, CT_vc, bias=True)[0][1] / np.var(CT_vc)
#                 print("beta = ", beta)
            else:
                beta = 1
#                 print("beta = ", beta)

            Ct_vc_exact = BSM(S=S, K=K, T=T[-1], sigma=sigma, r=r, y=y, t=t, kind='c', ret='option')
            CT1 = CT - CT_vc * beta + beta * Ct_vc_exact / discount_factor.prod()

            # Calculate call value using geometric call as control variate
            # First calculate the exact value of gc
            if Ta == T[-1]:  # averaging is the whole period
                r_eff = r
                y_eff = (r * (T[-1] - delta_t[-1]) / 2 + (y + (1 - 1 / m) * (sigma ** 2) / 6) * (
                         T[-1] + delta_t[-1]) / 2) / T[-1]
                sigma_eff = sigma / np.sqrt(3) * np.sqrt((1 + 1 / (2 * m)) * (T[-1] + delta_t[-1])) / np.sqrt(T[-1])
                # print("r_eff = ", r_eff)
                # print("y_eff = ", y_eff)
                # print("sigma_eff = ", sigma_eff)
                Ct_gc_exact = BSM(S=S, K=K, T=T[-1], r=r_eff, y=y_eff, sigma=sigma_eff)  # Valuation at t = 0

            elif Ta < T[-1]:  # averaging period is less than whole period
                r_eff = r
                y_eff = y
                m = T.shape[0] - 1
                k = int(T[0] * m / Ta)
                sigma_eff = np.sqrt((sigma**2)*T[0] +
                                    (1+1/(2*m)) * (sigma**2) * (T[-1]-T[0]+delta_t[-1]) / 3) / np.sqrt(T[-1])
                S_eff = S * np.exp(0.5*mu_tilde*(T[-1]+T[0]) + 0.5*sigma_eff**2) * discount_factor.prod()
                # print("r_eff = ", r_eff)
                # print("y_eff = ", y_eff)
                # print("sigma_eff = ", sigma_eff)
                # print("m = ", m)
                # print("k = ", k)
                # print("S_eff = ", S_eff)
                Ct_gc_exact = BSM(S=S, K=K, T=T[-1], r=r_eff, y=y_eff, sigma=sigma_eff)  # Valuation at t = 0

            # print("For m = {0:d}, geometric call is {1:.6f}".format(m, Ct_gc_exact))

            if control_loading:
                beta = np.cov(CT, CT_gc, bias=True)[0][1] / np.var(CT_gc)
#                 print("beta = ", beta)
            else:
                beta = 1
#                 print("beta = ", beta)

            CT2 = CT - CT_gc * beta + beta * Ct_gc_exact / discount_factor.prod()

            if control=='gc':
                CT = CT2
            elif control=='vc':
                CT = CT1
            elif control=='gc,vc' or control=='vc,gc':
                # import statsmodels.api as sm
                lm = sm.OLS(CT, np.c_[CT_gc, CT_vc], cov_type='HC0').fit()
#                 print("beta1 and beta2:", lm.params)
                # print(lm.summary())
                beta = lm.params
                CT = CT - CT_gc * beta[0] + beta[0] * Ct_gc_exact / discount_factor.prod() \
                        - CT_vc * beta[1] + beta[1] * Ct_vc_exact / discount_factor.prod()

    elif kind.lower() == 'gc':  # Asian geometric call
        asian_idx = (T2[1:] > T2[-1] - Ta)  # remove the 0 at the beginning
        ma = asian_idx.sum()
        if antithetic:
            STa_1 = St_1[:,asian_idx].prod(axis=1) ** (1. / ma)  # geometric mean
            STa_2 = St_2[:,asian_idx].prod(axis=1) ** (1. / ma)  # geometric mean
            CT_1 = np.amax(np.c_[STa_1 - K, np.zeros(STa_1.shape)], axis=1)  # Standard European Call payoff
            CT_2 = np.amax(np.c_[STa_2 - K, np.zeros(STa_2.shape)], axis=1)  # Standard European Call payoff, antithetic
            CT = (CT_1 + CT_2) / 2
        else:
            STa = St[:, asian_idx].prod(axis=1) ** (1. / ma)  # geometric mean
            CT = np.amax(np.c_[STa - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff

    elif kind.lower() == 'p':
        if flag_power_option == False:  # regular option
            if antithetic:
                CT_1 = np.amax(np.c_[K - ST_1, np.zeros(ST.shape)], axis=1)  # Standard European Put payoff
                CT_2 = np.amax(np.c_[K - ST_2, np.zeros(ST.shape)], axis=1)  # Standard European Put payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[K - ST, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
        else:  # general power option
            if antithetic:
                CT_1 = np.amax(np.c_[(K ** alpha_k - ST_1 ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Put payoff
                CT_2 = np.amax(np.c_[(K ** alpha_k - ST_2 ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                               axis=1)  # Standard European Put payoff, antithetic
                CT = (CT_1 + CT_2) / 2
            else:
                CT = np.amax(np.c_[(K ** alpha_k - ST ** alpha_s) ** alpha_tot, np.zeros(ST.shape)],
                             axis=1)  # Standard European Call payoff

    elif kind.lower() == 'logcall':
        if antithetic == True:
            CT_1 = np.amax(np.c_[K * np.log(ST_1 / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff
            CT_2 = np.amax(np.c_[K * np.log(ST_2 / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff
            CT = (CT_1 + CT_2) / 2
        else:  # antithetic==False
            CT = np.amax(np.c_[K * np.log(ST / K), np.zeros(ST.shape)], axis=1)  # LogCall payoff

        if controlvariate == True:
            if antithetic:
                CT_1_vc = np.amax(np.c_[ST_1 - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff
                CT_2_vc = np.amax(np.c_[ST_2 - K, np.zeros(ST.shape)],
                                  axis=1)  # Standard European Call payoff, antithetic
                CT_vc = (CT_1_vc + CT_2_vc) / 2
            else:  # antithetic==False
                CT_vc = np.amax(np.c_[ST - K, np.zeros(ST.shape)], axis=1)  # Standard European Call payoff

            CT = CT - CT_vc + BSM(S=S, K=K, T=T[-1], sigma=sigma, r=r, y=y, t=t, kind='c', ret='option') / \
                              discount_factor.prod()

    ## C represents general derivitive with a payoff function
    # Time T payoff stats
    CT_mean = CT.mean()
    CT_std = CT.std()
    CT_mean_std = CT_std * one_over_sqrt_n

    # Time t payoff stats
    Ct_mean = CT_mean * discount_factor.prod()
    Ct_mean_std = CT_mean_std * discount_factor.prod()

    if control_loading:
        return Ct_mean, Ct_mean_std, CT_mean, CT_mean_std, ST_mean, ST_mean_std, beta
    else:
        return Ct_mean, Ct_mean_std, CT_mean, CT_mean_std, ST_mean, ST_mean_std


def test_msmc():
    # n_list = [1000, 10000, 100000, 1000000, 10000000]
    n_list = [1000, 10000, 100000]
    m_list = [1, 2, 6, 12, 24, 120]

    results = []
    for m in m_list:
        if m != 12:
            pass
        for n in n_list:
            print("\n\nm = {0:d}\t n = {1:d}".format(m, n))

            #         output_bbmc = optionMC_multi_step(S=100, K=100, T=np.linspace(0.,1.,m+1), sigma=0.20, r=0.01,
            #                                            y=0.02, t=0., kind='ac', Ta=1.0, n=n, antithetic=False,
            #                                            controlvariate=False,
            #                                            moment_matching=True, matching_style='log', seed=402)

            #         output_atmc = optionMC_multi_step(S=100, K=100, T=np.linspace(0.,1.,m+1), sigma=0.20, r=0.01,
            #                                            y=0.02, t=0., kind='ac', Ta=1.0, n=n, antithetic=True,
            #                                            controlvariate=False,
            #                                            moment_matching=True, matching_style='log', seed=402)

            #         output_cv_vc = optionMC_multi_step(S=100, K=100, T=np.linspace(0.,1.,m+1), sigma=0.20, r=0.01,
            #                                            y=0.02, t=0., kind='ac', Ta=1.0, n=n, antithetic=True,
            #                                            controlvariate=True, control='vc', control_loading=True,
            #                                            moment_matching=True, matching_style='log', seed=402)

            #         output_cv_gc = optionMC_multi_step(S=100, K=100, T=np.linspace(0.,1.,m+1), sigma=0.20, r=0.01,
            #                                            y=0.02, t=0., kind='ac', Ta=1.0, n=n, antithetic=True,
            #                                            controlvariate=True, control='gc', control_loading=True,
            #                                            moment_matching=True, matching_style='log', seed=402)

            #         output_cv_gcvc = optionMC_multi_step(S=100, K=100, T=np.linspace(0.,1.,m+1), sigma=0.20, r=0.01,
            #                                            y=0.02, t=0., kind='ac', Ta=1.0, n=n, antithetic=True,
            #                                            controlvariate=True, control='gc,vc', control_loading=True,
            #                                            moment_matching=True, matching_style='log', seed=402)


            output_bbmc = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135, r=0.0091,
                                              y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=False,
                                              controlvariate=False,
                                              moment_matching=True, matching_style='log', seed=402)

            output_atmc = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135, r=0.0091,
                                              y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                              controlvariate=False,
                                              moment_matching=True, matching_style='log', seed=402)

            output_cv_vc = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135, r=0.0091,
                                               y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                               controlvariate=True, control='vc', control_loading=False,
                                               moment_matching=True, matching_style='log', seed=402)

            output_cv_vc_beta = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135,
                                                    r=0.0091,
                                                    y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                                    controlvariate=True, control='vc', control_loading=True,
                                                    moment_matching=True, matching_style='log', seed=402)

            output_cv_gc = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135, r=0.0091,
                                               y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                               controlvariate=True, control='gc', control_loading=False,
                                               moment_matching=True, matching_style='log', seed=402)

            output_cv_gc_beta = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135,
                                                    r=0.0091,
                                                    y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                                    controlvariate=True, control='gc', control_loading=True,
                                                    moment_matching=True, matching_style='log', seed=402)

            output_cv_gcvc = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135, r=0.0091,
                                                 y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                                 controlvariate=True, control='gc,vc', control_loading=False,
                                                 moment_matching=True, matching_style='log', seed=402)

            output_cv_gcvc_beta = optionMC_multi_step(S=2450., K=2450., T=np.linspace(4.5, 5., m + 1), sigma=0.135,
                                                      r=0.0091,
                                                      y=0.0195, t=0., kind='ac', Ta=0.5, n=n, antithetic=True,
                                                      controlvariate=True, control='gc,vc', control_loading=True,
                                                      moment_matching=True, matching_style='log', seed=402)

            #         print("Single step MC:")
            #         print("Ct: {0:.6f}+/-{1:.6f}".format(output_single[0], output_single[1]))
            #         print("CT: {0:.6f}+/-{1:.6f}".format(output_single[2], output_single[3]))
            #         print("ST: {0:.6f}+/-{1:.6f}".format(output_single[4], output_single[5]))

            print("\nMulti-step MC:")
            print("Ct: {0:.6f}+/-{1:.6f}".format(output_cv_gcvc[0], output_cv_gcvc[1]))
            print("CT: {0:.6f}+/-{1:.6f}".format(output_cv_gcvc[2], output_cv_gcvc[3]))
            print("ST: {0:.6f}+/-{1:.6f}".format(output_cv_gcvc[4], output_cv_gcvc[5]))

            record = [m, n]
            record.extend([output_bbmc[0], output_bbmc[1],
                           output_atmc[0], output_atmc[1],
                           output_cv_vc[0], output_cv_vc[1],
                           output_cv_vc_beta[0], output_cv_vc_beta[1], output_cv_vc_beta[6],
                           output_cv_gc[0], output_cv_gc[1],
                           output_cv_gc_beta[0], output_cv_gc_beta[1], output_cv_gc_beta[6],
                           output_cv_gcvc[0], output_cv_gcvc[1],
                           output_cv_gcvc_beta[0], output_cv_gcvc_beta[1], output_cv_gcvc_beta[6][0],
                           output_cv_gcvc_beta[6][1]
                           ])
            results.append(record)

    res = pd.DataFrame(results, columns=['m', 'n', 'Ct_mean(BBMC)', 'Ct_std(BBMC)',
                                         'Ct_mean(ATVMC)', 'Ct_std(ATVMC)',
                                         'Ct_mean(CVMC_VC)', 'Ct_std(CVMC_VC)',
                                         'Ct_mean(CVMC_VC_beta)', 'Ct_std(CVMC_VC_beta)', 'beta',
                                         'Ct_mean(CVMC_GC)', 'Ct_std(CVMC_GC)',
                                         'Ct_mean(CVMC_GC_beta)', 'Ct_std(CVMC_GC_beta)', 'beta'
                                                                                          'Ct_mean(CVMC_GCVC)',
                                         'Ct_std(CVMC_GCVC)',
                                         'Ct_mean(CVMC_GCVC_beta)', 'Ct_std(CVMC_GCVC_beta)', 'beta1', 'beta2'
                                         ])
    return res


def test2():
    C_bsm = BSM(S=2450, K=2450, T=5, sigma=0.135, r=0.0091, y=0.0195, t=0, kind='c', ret='option')

    output_single = optionMC_one_step(S=2450, K=2450, T=5, sigma=0.135, r=0.0091, y=0.0195, t=0, kind='c',
                                      n=2 ** 20, antithetic=True, controlvariate=False,
                                      moment_matching=True, matching_style='log', seed=402)

    output_cv_vc = optionMC_multi_step(S=2450, K=2450, T=np.linspace(4.5, 5, 6 + 1), sigma=0.135, r=0.0091,
                                       y=0.0195, t=0, kind='ac', Ta=0.5, n=10 ** 6, antithetic=True,
                                       controlvariate=True, control='vc',
                                       moment_matching=True, matching_style='log', seed=402)

    output_cv_gc = optionMC_multi_step(S=2450, K=2450, T=np.linspace(4.5, 5, 6 + 1), sigma=0.135, r=0.0091,
                                       y=0.0195, t=0, kind='ac', Ta=0.5, n=10 ** 6, antithetic=True,
                                       controlvariate=True, control='gc',
                                       moment_matching=True, matching_style='log', seed=402)

    print("BSM solution: {0:.6f}\n".format(C_bsm))

    print("Single step MC:")
    print("Ct: {0:.6f}+/-{1:.6f}".format(output_single[0], output_single[1]))
    print("CT: {0:.6f}+/-{1:.6f}".format(output_single[2], output_single[3]))
    print("ST: {0:.6f}+/-{1:.6f}".format(output_single[4], output_single[5]))

    print("\nMulti-step MC (CV-VC):")
    print("Ct: {0:.6f}+/-{1:.6f}".format(output_cv_vc[0], output_cv_vc[1]))
    print("CT: {0:.6f}+/-{1:.6f}".format(output_cv_vc[2], output_cv_vc[3]))
    print("ST: {0:.6f}+/-{1:.6f}".format(output_cv_vc[4], output_cv_vc[5]))

    print("\nMulti-step MC (CV-GC:")
    print("Ct: {0:.6f}+/-{1:.6f}".format(output_cv_gc[0], output_cv_gc[1]))
    print("CT: {0:.6f}+/-{1:.6f}".format(output_cv_gc[2], output_cv_gc[3]))
    print("ST: {0:.6f}+/-{1:.6f}".format(output_cv_gc[4], output_cv_gc[5]))


def main():
    # grouped_test2()

    # test2()
    res = test_msmc()
    # res.to_csv("Q2B.csv")
    # plt.show()


if __name__ == '__main__':
    main()
