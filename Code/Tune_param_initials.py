from basic_functions import *

def experiment_tune_initials(nx, objective, Jacobian, sigma, epsilon_range, alpha_range, N_sample=10,
                             method_str='Newton-CG'):
    """
        Tune hyperparameters alpha and epsilon with both types of initial vectors and fixed sigma using Scipy solver.
        This is to compare the performance of both types of initial vectors on the same images.

        input:
            nx - x-dim of the image
            objective - objective function
            Jacobian - gradient of objective function
            sigma - noise level
            epsilon_range - values of epsilon to be tuned
            alpha_range - values of alpha to be tuned
            method_str - type of solver, usually Newton-CG
            N_sample - number of training images
            method_str - type of solve, default 'Newton-CG'

        output:
            mean_result_local - average on the results from N_sample images, w.r.t each combination of parameters,
                                using local min as initial vector
            mean_result_zero - average on the results from N_sample images, w.r.t each combination of parameters,
                                using zero vector as initial vector
            full_rmse_local - RMSE from N_sample images, using local min as initial vector
            full_rmse_zero - RMSE from N_sample images, using zero vector as initial vector
            full_dice_local - Dice score from N_sample images, using local min as initial vector
            full_dice_zero - Dice score from N_sample images, using zero vector as initial vector

    """
    # graph Laplacian matrix
    L = getLaplacian_Sparse(nx)

    # initial
    initial = np.zeros(nx ** 2)

    ground_truth = np.array([])
    noisy_image = np.array([])

    full_rmse_local = np.array([])
    full_dice_local = np.array([])

    full_rmse_zero = np.array([])
    full_dice_zero = np.array([])

    for i in range(0, N_sample):
        # ground truth
        u = getPhantom(nx)
        ground_truth = np.append(ground_truth, u.ravel())

        # noisy image
        v = getNoisyImage(u, sigma)
        noisy_image = np.append(noisy_image, v.ravel())

        start1 = datetime.now()
        out1 = tune_params_scipy_diffInitial(objective, Jacobian, method_str, sigma,
                                             epsilon_range, alpha_range, u, v, L)
        end1 = datetime.now()
        print("Tuning this image using local min used time: {}".format(end1 - start1), end="\n")

        start2 = datetime.now()
        out2 = tune_params_scipy(objective, Jacobian, initial, method_str, sigma,
                                 epsilon_range, alpha_range, u, v, L)
        end2 = datetime.now()
        print("Tuning this image using zero vector used time: {}".format(end2 - start2), end="\n")


        full_rmse_local = np.append(full_rmse_local, out1[1][:, 3])
        full_dice_local = np.append(full_dice_local, out1[1][:, 4])

        full_rmse_zero = np.append(full_rmse_zero, out2[1][:, 3])
        full_dice_zero = np.append(full_dice_zero, out2[1][:, 4])

    ground_truth = ground_truth.reshape((-1, N_sample), order='F')
    noisy_image = noisy_image.reshape((-1, N_sample), order='F')

    full_rmse_local = full_rmse_local.reshape((-1, N_sample), order='F')
    full_dice_local = full_dice_local.reshape((-1, N_sample), order='F')

    full_rmse_zero = full_rmse_zero.reshape((-1, N_sample), order='F')
    full_dice_zero = full_dice_zero.reshape((-1, N_sample), order='F')

    paramlist = list(itertools.product([sigma], alpha_range, epsilon_range))

    full_rmse_local = np.concatenate((paramlist, full_rmse_local), axis=1)
    full_dice_local = np.concatenate((paramlist, full_dice_local), axis=1)

    full_rmse_zero = np.concatenate((paramlist, full_rmse_zero), axis=1)
    full_dice_zero = np.concatenate((paramlist, full_dice_zero), axis=1)

    mean_result_local = np.column_stack(
        (paramlist, np.mean(full_rmse_local[:, 3:], axis=1), np.mean(full_dice_local[:, 3:], axis=1)))
    mean_result_zero = np.column_stack(
        (paramlist, np.mean(full_rmse_zero[:, 3:], axis=1), np.mean(full_dice_zero[:, 3:], axis=1)))


    row_ind_rmse_local = mean_result_local[:, 3].argmin()
    print("Best combination according to rmse using local min is: ", mean_result_local[row_ind_rmse_local, :])
    row_ind_dice_local = mean_result_local[:, 4].argmax()
    print("Best combination according to dice score using local min is: ", mean_result_local[row_ind_dice_local, :])

    row_ind_rmse_zero = mean_result_zero[:, 3].argmin()
    print("Best combination according to rmse using zero vector is: ", mean_result_zero[row_ind_rmse_zero, :])
    row_ind_dice_zero = mean_result_zero[:, 4].argmax()
    print("Best combination according to dice score using zero vector is: ", mean_result_zero[row_ind_dice_zero, :])

    np.savetxt("mean_result_local.csv", mean_result_local, delimiter=",")
    np.savetxt("mean_result_zero.csv", mean_result_zero, delimiter=",")
    np.savetxt("full_rmse_local.csv", full_rmse_local, delimiter=",")
    np.savetxt("full_rmse_zero.csv", full_rmse_zero, delimiter=",")
    np.savetxt("full_dice_local.csv", full_dice_local, delimiter=",")
    np.savetxt("full_dice_zero.csv", full_dice_zero, delimiter=",")

    return mean_result_local, mean_result_zero, full_rmse_local, full_rmse_zero, full_dice_local, full_dice_zero


if __name__ == '__main__':
    nx = 128
    sigma = 1
    epsilon_range = np.linspace(0.08, 0.01, 7, endpoint=False)
    alpha_range = np.linspace(0.5, 0, 5, endpoint=False)
    out = experiment_tune_initials(nx, objective_function, objective_function_gradient, sigma, epsilon_range,
                                   alpha_range, N_sample=10, method_str='Newton-CG')
