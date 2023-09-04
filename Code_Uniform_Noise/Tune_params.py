from basic_functions import *

def experiment_tune(nx, objective, Jacobian, sigma, epsilon_range, alpha_range, N_sample=10, method_str='Newton-CG',
                    diff_initial=True):
    """
        Tune hyperparameters alpha and epsilon with specified initial vector and fixed sigma using Scipy solver.

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
            diff_initial - True: using local min (v-dependent) as initial vector;
                           False: using zero vector as initial vector

        output:
            mean_result - average on the results from N_sample images, w.r.t each combination of parameters
            full_rmse - RMSE from N_sample images
            full_dice - Dice score from N_sample images

    """
    # graph Laplacian matrix
    L = getLaplacian_Sparse(nx)

    # initial
    initial = np.zeros(nx ** 2)

    full_rmse = np.array([])
    full_dice = np.array([])
    ground_truth = np.array([])
    noisy_image = np.array([])

    for i in range(0, N_sample):
        # ground truth
        u = getPhantom(nx)
        ground_truth = np.append(ground_truth, u.ravel())

        # noisy image
        v = getNoisyImage(u, sigma)
        noisy_image = np.append(noisy_image, v.ravel())

        start = datetime.now()
        if diff_initial == True:
            out = tune_params_scipy_diffInitial(objective, Jacobian, method_str, sigma,
                                                epsilon_range, alpha_range, u, v, L)
        else:
            out = tune_params_scipy(objective, Jacobian, initial, method_str, sigma,
                                    epsilon_range, alpha_range, u, v, L)

        end = datetime.now()
        print("Tuning this image using scipy used time: {}".format(end - start), end="\n")

        # np.savetxt('result_rmse_scipy_sigma'+str(sigma)+'_'+str(i)+'.csv', out[1], delimiter=',')
        # np.savetxt('result_image_scipy_sigma'+str(sigma)+'_'+str(i) + '.csv', out[0], delimiter=',')
        
        full_rmse = np.append(full_rmse, out[1][:, 3])
        full_dice = np.append(full_dice, out[1][:, 4])

    ground_truth = ground_truth.reshape((-1, N_sample), order='F')
    noisy_image = noisy_image.reshape((-1, N_sample), order='F')
    full_rmse = full_rmse.reshape((-1, N_sample), order='F')
    full_dice = full_dice.reshape((-1, N_sample), order='F')

    paramlist = list(itertools.product([sigma], alpha_range, epsilon_range))
    full_rmse = np.concatenate((paramlist, full_rmse), axis=1)
    full_dice = np.concatenate((paramlist, full_dice), axis=1)

    mean_result = np.column_stack((paramlist, np.mean(full_rmse[:, 3:], axis=1), np.mean(full_dice[:, 3:], axis=1)))

    row_ind_rmse = mean_result[:, 3].argmin()
    print("Best combination according to rmse is: ", mean_result[row_ind_rmse, :])
    row_ind_dice = mean_result[:, 4].argmax()
    print("Best combination according to dice score is: ", mean_result[row_ind_dice, :])

    # np.savetxt("ground_truth_sigma" +  str(sigma) + ".csv", ground_truth, delimiter=",")
    # np.savetxt("noisy_image_sigma" +  str(sigma) + ".csv", noisy_image, delimiter=",")
    # np.savetxt("full_rmse_sigma" + str(sigma) + ".csv", full_rmse, delimiter=",")
    # np.savetxt("full_dice_sigma" + str(sigma) + ".csv", full_dice, delimiter=",")
    # np.savetxt("mean_result_sigma" + str(sigma) + ".csv", mean_result, delimiter=",")

    return mean_result, full_rmse, full_dice


if __name__ == '__main__':
    # Example on using zero vector as initial vector
    nx = 128
    sigma = 1
    epsilon_range = np.linspace(0.1, 0, 10, endpoint=False)
    alpha_range = np.linspace(0.5, 0, 5, endpoint=False)
    out = experiment_tune(nx, objective_function, objective_function_gradient, sigma,
                    epsilon_range, alpha_range, N_sample=10, method_str='Newton-CG', diff_initial=False)

    # For sigma=0.1
    # sigma = 0.1
    # alpha_range = [0.4, 0.3, 0.2, 0.1, 0.05]
    # epsilon_range = [0.3,0.2,0.1,0.07,0.05,0.04,0.03,0.02,0.01]

    # For sigma=0.5
    # sigma = 0.5
    # alpha_range = np.linspace(0.5, 0, 5, endpoint=False)
    # epsilon_range = [0.3,0.2,0.15,0.1,0.09,0.08,0.07,0.06,0.05,0.04]
