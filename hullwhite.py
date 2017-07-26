
import pandas as pd
import numpy as np
import scipy.integrate as it
import scipy.linalg as sla
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.dates

plt.style.use('ggplot')
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.color'] = 'blue'
mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['axes.facecolor'] = '#f0f0f0'
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['axes.labelpad'] = 0.0
mpl.rcParams['axes.xmargin'] = 0.05  # x margin.  See `axes.Axes.margins`
mpl.rcParams['axes.ymargin'] = 0.05  # y margin See `axes.Axes.margins`
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['figure.subplot.left'] = 0.08
mpl.rcParams['figure.subplot.right'] = 0.95
mpl.rcParams['figure.subplot.bottom'] = 0.07

# A specific discaount curve for testing and calculating theta
def Z(T):
    """5th order polynomial fit of a specific discount curve for testing purpose"""
    # log(𝑍(𝑇))=𝑎𝑇+𝑏𝑇2+𝑐𝑇3+𝑑𝑇4+𝑒𝑇5
    a = -3.26115568e-02
    b = -1.07789195e-03
    c = -1.98829878e-05
    d = 2.85325674e-06
    e = -4.78160451e-08
    Z = np.exp(a*T + b*T**2 + c*T**3 + d*T**4 + e*T**5)
    return Z


def get_spot(T):
    """Calculate the spot rate for maturity T"""
    # log(𝑍(𝑇))=𝑎𝑇+𝑏𝑇2+𝑐𝑇3+𝑑𝑇4+𝑒𝑇5
    a = -3.26115568e-02
    b = -1.07789195e-03
    c = -1.98829878e-05
    d = 2.85325674e-06
    e = -4.78160451e-08
    logZ = a * T + b * T ** 2 + c * T ** 3 + d * T ** 4 + e * T ** 5
    spot = -logZ / T
    spot[0] = -a
    return spot


def get_forward(T):
    """Calculate the instantaneous forward rates f(0, T) from polynomial fitted discount curve """
    a = -3.26115568e-02
    b = -1.07789195e-03
    c = -1.98829878e-05
    d = 2.85325674e-06
    e = -4.78160451e-08
    forward = -(a + 2*b*T + 3*c*T**2 + 4*d*T**3 + 5*e*T**4)
    return forward


def get_dforward_to_dT(T):
    """Calculate the derivative of instantaneous forward rate to T"""
    a = -3.26115568e-02
    b = -1.07789195e-03
    c = -1.98829878e-05
    d = 2.85325674e-06
    e = -4.78160451e-08
    df = -(2*b + 2*3*c*T**1 + 3*4*d*T**2 + 4*5*e*T**3)
    return df


def get_par_yield(T, r_spot, Tnew=[], k=2, t=0, compound=0):
    """Calculate the par yield curve"""
    from scipy import interpolate

    Tcopy = T.copy()
    if Tcopy[0] == 0:
        Tcopy[0] = 0.00001

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
        if compound == 0:
            disc_i = np.exp(-(T2[i] - t_coupon_i) * r_spot_i)  # Use continuous compounding for spot rate
        else:
            disc_i = (1 + r_spot_i / k) ** (-k * (T2[i] - t_coupon_i))  # Use semi-annual compounding for spot rate
        # print('Discount at coupon time', disc_i)
        if compound == 0:
            r_par2[i] = 2 * (
            1 - np.exp(-(T2[i] - t) * r_spot2[i])) / disc_i.sum()  # Use continuous compounding for spot rate
        else:
            r_par2[i] = 2 * (
            1 - (1 + r_spot2[i] / k) ** (-k * (T2[i] - t))) / disc_i.sum()  # Use semi-annual compounding for spot rate

    # Need to compensate for accured interest
    r_par2 = (np.ceil(T2 / 0.5) / 2) / T2 * r_par2

    # Interpolate back to the original time grid (or otherwise specified)
    f_x = interpolate.interp1d(T2, r_par2, kind='cubic', fill_value='extrapolate')
    if len(Tnew) == 0:  # Tnew not specified
        r_par = f_x(T)
    else:
        r_par = f_x(Tnew)
    return r_par


def get_theta(sigma, kappa, T):
    """Calculate theta (variable drift matched to today's yield curve) as a function of T (maturity)"""
    theta = get_dforward_to_dT(T) + kappa*get_forward(T) + sigma**2/(2*kappa)*(1-np.exp(-2*kappa*T))
    return theta


def test_theta():
    Tmax = 30
    n = 60
    t = 0
    dt = Tmax / n
    kappa = 0.15
    sigma = 0.015
    T = np.linspace(0, Tmax, n + 1)
    forward = get_forward(T)
    theta = get_theta(kappa, sigma, T)

    fig, ax = plt.subplots()
    ax.plot(T, forward, 'r', label='forward rate')
    ax.plot(T, theta, 'b', label='theta')
    plt.title(r'$\theta(T)$')
    plt.xlabel('Maturity')
    plt.ylabel('')
    plt.legend(loc=0)
    plt.draw()
    print(theta)


def draw_uniform(size=5, low=0., high=1., seed=42):
    """Draw uniformly distributed random numbers"""
    np.random.seed(seed)
    return np.random.uniform(low, high, size)


def draw_normal(size=1, mu=0., sigma=1., seed=42):
    """Draw normally distribtued random numbers"""
    np.random.seed(seed)
    return np.random.normal(loc=mu, scale=sigma, size=size)


# Implementation of Hull-White Model
def get_B_HW(kappa, t, T):
    """
    Calculate B(t, T)
    T can be vectorized
    """
    Bt_T = (1 - np.exp(-kappa * (T - t))) / kappa
    return Bt_T


def get_A_HW(kappa, sigma, t, T):
    """
    Calculate A(t, T)
    T can be vectorized
    """
    if type(T) != np.ndarray:  # non-vectorized, T is a float
        At_T1 = - it.quad(lambda x: get_B_HW(kappa, x, T) * get_theta(kappa, sigma, x), t, T)[0]
    else:  # T is a vectorized 1-D array
        At_T1 = np.zeros(T.shape)
        for i in range(T.shape[0]):
            Ti = T[i]
            At_T1[i] = - it.quad(lambda x: get_B_HW(kappa, x, Ti) * get_theta(kappa, sigma, x), t, Ti)[0]

    At_T2 = sigma ** 2 / (2 * kappa ** 2) * (
    T - t + (1 - np.exp(-2 * kappa * (T - t))) / (2 * kappa) - 2 * get_B_HW(kappa, t, T))
    At_T = At_T1 + At_T2
    return At_T


def calc_ZCB_HW(kappa, sigma, r, t, T):
    Bt_T = get_B_HW(kappa, t, T)
    At_T = get_A_HW(kappa, sigma, t, T)
    ZCB = np.exp(At_T - Bt_T * r)
    return ZCB


def valueHullWhite_by_MC(func_payoff, T, t=0, kappa=0.15, sigma=0.015, r=0.03, n_steps=100, n_paths=500,
                         seed=405, plot=False):
    """Monte-Carlo"""
    delta_t = (T - t) / n_steps
    Tspace = np.linspace(t, T, n_steps)
    theta = get_theta(sigma, kappa, Tspace)

    # Draw standard normal shocks
    dW = draw_normal((n_steps, n_paths), seed=seed)

    r_paths = np.zeros((n_steps + 1, n_paths))
    ra_paths = np.zeros((n_steps + 1, n_paths))

    for i in range(n_steps + 1):
        if i == 0:
            r_paths[0, :] = r
            ra_paths[0, :] = r
        else:  # i from 1 to n_steps
            delta_r = (theta[i - 1] - kappa * r_paths[i - 1, :]) * delta_t + sigma * np.sqrt(delta_t) * dW[i - 1, :]
            r_paths[i, :] = r_paths[i - 1, :] + delta_r
            # Anti-thetic paths
            delta_ra = (theta[i - 1] - kappa * ra_paths[i - 1, :]) * delta_t - sigma * np.sqrt(delta_t) * dW[i - 1, :]
            ra_paths[i, :] = ra_paths[i - 1, :] + delta_ra

    if plot:
        fig, ax = plt.subplots()
        _ = plt.plot(Tspace, theta)
        plt.title(r'$\theta(t)$')
        fig, ax = plt.subplots()
        _ = plt.plot(r_paths, alpha=0.2)
        plt.title(r'Short rate paths')

    payoff = func_payoff(r_paths, T)
    payoff_at = func_payoff(ra_paths, T)
    print("---Payoff---\nOriginal")
    print("Mean: {0:.4f} \tStd: {1:.4f}".format(payoff.mean(), payoff.std() / np.sqrt(n_paths)))
    print("\nAntithetic")
    print("Mean: {0:.4f} \tStd: {1:.4f}".format(payoff_at.mean(), payoff_at.std() / np.sqrt(n_paths)))

    payoff_pv = payoff * np.exp(-r_paths.sum(axis=0) * delta_t)
    payoff_at_pv = payoff_at * np.exp(-ra_paths.sum(axis=0) * delta_t)
    print("\n---Payoff PV ---\nOriginal")
    print("Mean: {0:.4f} \tStd: {1:.4f}".format(payoff_pv.mean(), payoff_pv.std() / np.sqrt(n_paths)))
    print("\nAntithetic")
    print("Mean: {0:.4f} \tStd: {1:.4f}".format(payoff_at_pv.mean(), payoff_at_pv.std() / np.sqrt(n_paths)))

    value_fut = 0.5 * (payoff + payoff_at)
    value_pv = 0.5 * (payoff_pv + payoff_at_pv)
    return value_pv.mean(), value_pv.std() / np.sqrt(n_paths), value_fut.mean(), value_fut.std() / np.sqrt(n_paths)


def solveHullWhite_by_FD(payoff, style='european', U=np.array([]), Tgrid=np.array([]), func_theta=get_theta, T=5, t=0,
                         kappa=0.15, sigma=0.015,
                         nt=150, nr=100, r_min=-0.05, r_max=0.30, plot=False, debug=False):
    """
    Solve Hull-Whtie PDE by finite difference method
    func_payoff:  Function that specify terminal payoff
    func_theta:   Function to get theta(T)
    """
    import scipy.linalg as sla
    from scipy.interpolate import interp1d

    T_space = np.linspace(0, T, nt + 1)
    r = np.linspace(r_min, r_max, nr + 1)
    theta = func_theta(kappa=kappa, sigma=sigma, T=T_space)
    #     theta = np.zeros(T_space.shape)

    delta_r = (r_max - r_min) / nr
    delta_t = (T - t) / nt

    V = np.zeros((nr + 1, nt + 1))

    if plot:
        fig, ax = plt.subplots()
        _ = plt.plot(T_space, theta)
        plt.title(r'$\theta(t)$')

    if debug:
        np.set_printoptions(precision=3)
        print("delta_r = ", delta_r)
        print("delta_t = ", delta_t)
        print("V.shape = ", V.shape)

    for j in range(nt, -1, -1):
        if j == nt:
            V[:, j] = payoff
        else:
            P = np.zeros((nr + 1, nr + 1))
            for i in range(0, nr + 1):
                if i == 0:
                    P[0, 0] = 1 + delta_t * (-sigma ** 2 / (2 * delta_r ** 2) +
                                             3 * (theta[j] - kappa * r[i]) / (4 * delta_r) +
                                             r[0])
                    P[0, 1] = delta_t * (sigma ** 2 / (delta_r ** 2) - (theta[j] - kappa * r[0]) / (2 * delta_r))
                    P[0, 2] = delta_t * (-sigma ** 2 / (2 * delta_r ** 2) - (theta[j] - kappa * r[0]) / (4 * delta_r))
                elif i == nr:
                    P[nr, nr - 2] = delta_t * (-sigma ** 2 / (2 * delta_r ** 2)
                                               + (theta[j] - kappa * r[nr]) / (4 * delta_r))
                    P[nr, nr - 1] = delta_t * (sigma ** 2 / (delta_r ** 2)
                                               + (theta[j] - kappa * r[nr]) / (2 * delta_r))
                    P[nr, nr] = 1 + delta_t * (-sigma ** 2 / (2 * delta_r ** 2)
                                               - 3 * (theta[j] - kappa * r[nr]) / (4 * delta_r) + r[nr])
                else:
                    P[i, i - 1] = delta_t * (-sigma ** 2 / (2 * delta_r ** 2)
                                             + (theta[j] - kappa * r[i]) / (2 * delta_r))
                    P[i, i] = 1 + delta_t * (sigma ** 2 / (delta_r ** 2) + r[i])
                    P[i, i + 1] = delta_t * (-sigma ** 2 / (2 * delta_r ** 2)
                                             - (theta[j] - kappa * r[i]) / (2 * delta_r))
            P_inv = sla.inv(P)
            V[:, j] = np.matmul(P_inv, V[:, j + 1])

            if style == 'american' or style == 'American':
                Uintp = np.zeros(r.shape)
                for k in range(len(r)):
                    func_u = interp1d(Tgrid, U[k, :], kind='cubic')
                    Uintp[k] = func_u(T_space[j])
                U_exercise = np.max([100 - Uintp, np.zeros(Uintp.shape)], axis=0)
                #                 if (U_exercise>V[:,j]).any():
                #                     print(U_exercise>V[:,j])
                V[:, j] = np.max([V[:, j], U_exercise], axis=0)
                #                 print("U\n", Uintp)

        if debug:
            np.set_printoptions(precision=3)
            print("\nj = ", j)
            if j != nt:
                print("\nP matrix\n", P)
                print("\nP_inv matrix\n", P_inv)
            print("\nV matrix\n", V)

    return V


def payoff_q2(r_paths, T):
    """ payoff function for Q2 """
    n_paths = r_paths.shape[1]
    n_steps = r_paths.shape[0] - 1

    n_steps_period = int(n_steps * 2 / T)
    n_steps_reduced = n_steps + 1 - n_steps_period

    r_period = np.zeros((n_steps_reduced, n_paths))

    for i in range(n_steps_reduced):
        r_period[i, :] = (r_paths[i:i + n_steps_period, :]).mean(axis=0)

    payoff = 1e7 * (r_period.max(axis=0) - r_period.min(axis=0))
    return payoff


def get_payoff_q3(VT):
    return np.ones(VT.shape)


def payoff_q4(r_paths, T):
    """payoff calculation for Q4"""
    assert T == 9
    n_paths = r_paths.shape[1]
    n_steps = r_paths.shape[0] - 1
    t_steps_per_year = int(n_steps / T)
    t_steps_half_year = int(t_steps_per_year / 2)
    delta_t = T / n_steps

    r_5to9 = r_paths[int(n_steps * 5 / 9):, :]

    payoff = np.zeros((1, r_paths.shape[1]))
    # Add all the discounted cash flows
    for i in range(8):
        if i == 7:
            cf = 102
        else:
            cf = 2
        payoff = payoff + cf * np.exp(- r_5to9[0:(i + 1) * t_steps_half_year, :].sum(axis=0) * delta_t)
    return payoff


def question4():
    """Question 4 testing Hull-White by MC"""
    print("Question 4")
    kappa = 0.15
    sigma = 0.015

    _, _, value_q4, value_q4_std = valueHullWhite_by_MC(payoff_q4, T=9, t=0, kappa=0.15, sigma=0.015,
                                                        r=get_forward(0), n_steps=9 * 12, n_paths=5000, seed=41,
                                                        plot=True)
    print("\nFutures price for payoff in Q4 is: ${0:.2f} +/- {1:.2f}".format(value_q4, value_q4_std))


def question3():
    """Qustion 3 testing Hull-White by Finite Difference"""
    print("Question 3")
    Tmax = 5
    n = 10
    t = 0
    dt = Tmax / n
    T = np.linspace(0, Tmax, n + 1)

    kappa = 0.15
    sigma = 0.015

    forward = get_forward(T)
    spot = get_spot(T)
    par = get_par_yield(T, spot, compound=0)
    theta = get_theta(sigma, kappa, T)

    table = pd.DataFrame({'Maturity': T, 'Spot Rate': spot, 'Forward Rate': forward,
                          'Par Yield': par})
    table.set_index('Maturity', inplace=True)
    table = table[['Forward Rate', 'Spot Rate', 'Par Yield']]
    print(table)

def main():
    question3()
    plt.show()


if __name__ == '__main__':
    main()
