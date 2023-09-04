from basic_functions import *
from matplotlib import colors

def experiment_test(nx, objective, Jacobian, params, N_sample=20, method_str='Newton-CG', diff_initial=True):
    # graph Laplacian matrix
    L = getLaplacian_Sparse(nx)

    # initial
    initial = np.zeros(nx ** 2)

    out_record = np.array([])

    for sigma, alpha, epsilon in params:
        for i in range(0, N_sample):
            # ground truth
            u = getPhantom(nx)
            # ground_truth = np.append(ground_truth, u.ravel())

            # noisy image
            v = getNoisyImage(u, sigma)
            # noisy_image = np.append(noisy_image, v.ravel())

            args = (v, L, alpha, epsilon)

            start = datetime.now()
            if diff_initial == True:

                x0 = get_uhat_numeric(alpha, epsilon, v)
                result = scipy.optimize.minimize(objective, x0, args, method=method_str, jac=Jacobian, tol=1e-06,
                                                 options={'maxiter': 200})

            else:
                result = scipy.optimize.minimize(objective, initial, args, method=method_str, jac=Jacobian, tol=1e-06,
                                                 options={'maxiter': 200})

            end = datetime.now()
            print("Noise level: ", sigma, ". Tuning image ", i, " using scipy used time: {}".format(end - start),
                  end="\n")

            r = getRMSE(u.ravel(), result.x)
            dice = getDiceScore(result.x, u.ravel())

            out_record = np.hstack((out_record, sigma, alpha, epsilon, r, dice, result.success))

    out_record = out_record.reshape(-1, 6)

    return out_record


def experiment_test_comparison(nx, objective, Jacobian, params_local, params_zero, N_sample=10, method_str='Newton-CG'):
    # graph Laplacian matrix
    L = getLaplacian_Sparse(nx)

    # initial
    initial = np.zeros(nx ** 2)

    out_record_local = np.array([])
    out_record_zero = np.array([])

    for (sigma1, alpha1, epsilon1), (sigma2, alpha2, epsilon2) in zip(params_local, params_zero):
        for i in range(0, N_sample):
            # ground truth
            u = getPhantom(nx)
            # ground_truth = np.append(ground_truth, u.ravel())

            # noisy image
            v = getNoisyImage(u, sigma1)
            # noisy_image = np.append(noisy_image, v.ravel())

            args1 = (v, L, alpha1, epsilon1)

            start = datetime.now()

            x0 = get_uhat_numeric(alpha1, epsilon1, v)
            result_local = scipy.optimize.minimize(objective, x0, args1, method=method_str, jac=Jacobian, tol=1e-06,
                                                   options={'maxiter': 200})

            end = datetime.now()
            print("Noise level: ", sigma1, ". Tuning image ", i, " using lcoal min used time: {}".format(end - start),
                  end="\n")

            args2 = (v, L, alpha2, epsilon2)
            start = datetime.now()
            result_zero = scipy.optimize.minimize(objective, initial, args2, method=method_str, jac=Jacobian, tol=1e-06,
                                                  options={'maxiter': 200})

            end = datetime.now()
            print("Noise level: ", sigma2, ". Tuning image ", i, " using zero vector used time: {}".format(end - start),
                  end="\n")

            r_local = getRMSE(u.ravel(), result_local.x)
            dice_local = getDiceScore(result_local.x, u.ravel())

            r_zero = getRMSE(u.ravel(), result_zero.x)
            dice_zero = getDiceScore(result_zero.x, u.ravel())

            out_record_local = np.hstack(
                (out_record_local, sigma1, alpha1, epsilon1, r_local, dice_local, result_local.success))
            out_record_zero = np.hstack(
                (out_record_zero, sigma2, alpha2, epsilon2, r_zero, dice_zero, result_zero.success))

    out_record_local = out_record_local.reshape(-1, 6)
    out_record_zero = out_record_zero.reshape(-1, 6)

    return out_record_local, out_record_zero


def plot_test_result(out_record, sigma_level=["0.1", "0.5", "1"], N_sample=20):
    x_value = np.repeat(sigma_level, N_sample)
    rmse = out_record[:, 3]
    dice = out_record[:, 4]
    group = out_record[:, 5]

    fig, ax = plt.subplots(1, 2)
    mycmap = colors.ListedColormap(['orangered', 'cornflowerblue'])
    ax[0].scatter(x_value, rmse, c=group, cmap=mycmap, s=15)
    ax[0].set_xlabel("Noise level " + r'$\sigma$')
    ax[0].set_ylabel("RMSE")
    ax[1].scatter(x_value, dice, c=group, cmap=mycmap, label="True", s=15)
    ax[1].set_xlabel("Noise level " + r'$\sigma$')
    ax[1].set_ylabel("Dice Score")
    ax[1].scatter([], [], marker="o", label="False", c='red', s=15)
    plt.legend(title="Converged")
    ax[1].legend(loc=(1.1, 0.5), title="Convergence")
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    nx = 128
    params_local = [[0.1, 0.1, 0.03], [0.5, 0.3, 0.08], [1, 0.2, 0.04]]
    params_zero = [[0.1, 0.1, 0.03], [0.5, 0.3, 0.08], [1, 0.3, 0.05]]
    out = experiment_test_comparison(nx, objective_function, objective_function_gradient, params_local, params_zero,
                                     N_sample=20, method_str='Newton-CG')
    plot_test_result(out[0], sigma_level=["0.1", "0.5", "1"], N_sample=20)
    plot_test_result(out[1], sigma_level=["0.1", "0.5", "1"], N_sample=20)
