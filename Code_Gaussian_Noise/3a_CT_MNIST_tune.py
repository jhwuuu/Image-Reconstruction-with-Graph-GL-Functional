from basic_functions import *
import GL_3SR
from torchvision.datasets import MNIST
import numpy as np
from sklearn.model_selection import train_test_split
import astra


def data_selection(X, y, digit, train_size=100, tune_size=50, test_size=20):

    digit = str(digit+1)
    idx = np.where(y==digit)[0]

    test_tune_size = tune_size + test_size
    X_train, X_test_tune, y_train, y_test_tune = train_test_split(
        X[idx], y[idx], train_size=train_size, test_size=test_tune_size, shuffle=True
    )

    X_tune, X_test, y_tune, y_test = train_test_split(
        X_test_tune, y_test_tune, train_size=tune_size, test_size=test_size, shuffle=True
    )

    return X_train, X_tune, X_test, y_train, y_tune, y_test



def get_binary_images(data, threshold):

    return np.where(data > threshold, 1, 0)

def get_noisy_binary_images(data, sigma):

    return data + np.random.default_rng().normal(0, sigma, (data.shape[0], data.shape[1]))


def fit_laplacian(data, beta, gamma, maxiter=100):

    Y = data.T
    N = Y.shape[0]
    gl3sr = GL_3SR.FGL_3SR(trace=N, beta=beta, alpha=gamma, maxit=maxiter, verbose=True, cv_crit=10e-12)
    gl3sr.fit(Y)

    X, H, lbd, err = gl3sr.get_coeffs()
    Lpred = X.dot(np.diag(lbd)).dot(X.T)
    Wpred = np.diag(np.diag(Lpred)) - Lpred
    Wpred = (Wpred + Wpred.T)/2
    Wpred *= (Wpred>np.mean(Wpred))

    return Lpred, Wpred, err, gl3sr


def sinograph_generator(ground_truth, sigma, angles, num_pixels_x=28, det_width=1, originToSource=100,
                        originToDetector=50):
    num_detectors = int(1.5 * num_pixels_x)

    vol_geom = astra.create_vol_geom(num_pixels_x, num_pixels_x)
    proj_geom = astra.create_proj_geom('fanflat', det_width, num_detectors, angles, originToSource, originToDetector)
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)
    # Get the projection matrix as a Scipy sparse matrix.
    K = astra.matrix.get(matrix_id)

    num_angles = angles.shape[0]
    f_delta = np.zeros((ground_truth.shape[0], num_angles * num_detectors))
    for i in range(ground_truth.shape[0]):
        gt = ground_truth[i, :]
        f_id, f = astra.create_sino(gt, proj_id)

        f_delta_2d = f + np.random.default_rng().normal(0, sigma, (f.shape[0],f.shape[1]))
        f_delta[i, :] = f_delta_2d.ravel()

    astra.data2d.delete(f_id)
    astra.projector.delete(proj_id)

    return K, f_delta


def K_generator(angles, num_pixels_x=28, det_width=1, originToSource=100,
                        originToDetector=50):
    num_detectors = int(1.5 * num_pixels_x)

    vol_geom = astra.create_vol_geom(num_pixels_x, num_pixels_x)
    proj_geom = astra.create_proj_geom('fanflat', det_width, num_detectors, angles, originToSource, originToDetector)
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)
    # Get the projection matrix as a Scipy sparse matrix.
    K = astra.matrix.get(matrix_id)

    astra.projector.delete(proj_id)

    return K

def objective_function_gradient_ct(u, f_delta, K, L,  epsilon):

    out =  np.transpose(K) @ ( K@u.ravel() - f_delta.ravel()) + L @ u.ravel() + 1 / epsilon * getWGradient(u.ravel())

    return out


def tune_ct_rec(initial, K, f_delta, L, sigma, alpha_range, epsilon_range, u_train, num_iter=30):
    nx = initial.shape[0]
    num_data = f_delta.shape[0]
    full_record = np.zeros((len(alpha_range) * len(epsilon_range), 11))
    for idx_param, (alpha, epsilon) in enumerate(itertools.product(alpha_range, epsilon_range)):
        result_rmse = np.array([])
        result_dice = np.array([])
        result_converged = np.array([])
        result_gradl2norm = np.array([])
        result_r_diff = np.array([])
        result_d_diff = np.array([])
        result_idx = np.array([])

        for idx_data in range(num_data):
            f = f_delta[idx_data, :]
            gt = u_train[idx_data, :].ravel()
            dice = np.array([])
            rmse = np.array([])
            l2norm = np.array([])
            converge = np.array([])
            all_vecs = np.zeros((num_iter, nx))
            u = initial.copy()
            v = u - alpha * K.transpose() @ (K @ u - f)
            out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, L, alpha, epsilon), method='Newton-CG',
                                          jac=objective_function_gradient, tol=1e-06,
                                          options={'maxiter': 100, 'return_all': True, 'disp': False})
            for iter in range(num_iter):
                u = out.x
                v = u - alpha * K.transpose() @ (K @ u - f)
                out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, L, alpha, epsilon),
                                              method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                              options={'maxiter': 100, 'return_all': True, 'disp': False})
                r = getRMSE(out.x, gt)
                d = getDiceScore(out.x, gt)
                conv = out.success
                # l = np.linalg.norm(objective_function_gradient(out.x, v, L, alpha, epsilon))

                # all_vecs[iter, :] = out.x
                dice = np.append(dice, d)
                rmse = np.append(rmse, r)
                # l2norm = np.append(l2norm, l)
                converge = np.append(converge, conv)

            l2norm = np.linalg.norm(objective_function_gradient_ct(out.x, f, K, L, epsilon))
            max_dice = np.max(dice)
            idx_d = np.where(dice == max_dice)[0]
            idx_r = np.argmin(rmse[idx_d])
            idx = idx_d[idx_r]

            last_rmse = rmse[-1]
            last_dice = dice[-1]
            r_diff = rmse[idx] - last_rmse
            d_diff = dice[idx] - last_dice

            result_idx = np.append(result_idx, idx)
            result_r_diff = np.append(result_r_diff, r_diff)
            result_d_diff = np.append(result_d_diff, d_diff)
            result_rmse = np.append(result_rmse, rmse[-1])
            result_dice = np.append(result_dice, dice[-1])
            result_converged = np.append(result_converged, converge[-1])
            result_gradl2norm = np.append(result_gradl2norm, l2norm)

        print("alpha = ", alpha, ", epsilon = ", epsilon, ", dice = ", np.round(result_dice, 3))
        full_record[idx_param, :] = [sigma, alpha, epsilon, np.mean(result_rmse), np.mean(result_dice),
                                     np.mean(result_gradl2norm), np.std(result_gradl2norm),
                                     np.mean(result_converged), np.mean(result_r_diff), np.mean(result_d_diff),
                                     np.mean(result_idx)]


    row_ind_rmse = full_record[:, 3].argmin()
    print("Best combination according to rmse is: ", full_record[row_ind_rmse, :])
    row_ind_dice = full_record[:, 4].argmax()
    print("Best combination according to dice score is: ", full_record[row_ind_dice, :])

    return full_record


def tune_ct_rec_Lpred(initial, K, f_delta, beta_range, alpha_range, epsilon_range, u_train, num_iter=30):
    nx = initial.shape[0]
    num_data = f_delta.shape[0]
    full_record = np.ones((1, 12))
    # for beta in beta_range:
    #     [Lpred, Wpred, err, gl3sr] = fit_laplacian(u_learnL.astype(float), gamma=0.001, beta=beta)
    #     Lpred_str = 'CT_Lpred_'+str(beta)+".csv"
    #     np.savetxt(Lpred_str, Lpred, delimiter=",")
    for beta in beta_range:
        Lpred_str = 'CT_Lpred_'+str(beta)+'.csv'
        Lpred = np.genfromtxt(Lpred_str, delimiter=',')

        record = np.zeros((len(alpha_range) * len(epsilon_range), 12))

        for idx_param, (alpha, epsilon) in enumerate(itertools.product(alpha_range, epsilon_range)):
            result_rmse = np.array([])
            result_dice = np.array([])
            result_converged = np.array([])
            result_gradl2norm = np.array([])
            result_r_diff = np.array([])
            result_d_diff = np.array([])
            result_idx = np.array([])

            for idx_data in range(num_data):
                f = f_delta[idx_data, :]
                gt = u_train[idx_data, :].ravel()
                dice = np.array([])
                rmse = np.array([])

                converge = np.array([])

                u = initial.copy()
                v = u - alpha * K.transpose() @ (K @ u - f)
                out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, Lpred, alpha, epsilon),
                                              method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                              options={'maxiter': 100, 'return_all': True, 'disp': False})
                for iter in range(num_iter):
                    u = out.x
                    v = u - alpha * K.transpose() @ (K @ u - f)
                    out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, Lpred, alpha, epsilon),
                                                  method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                                  options={'maxiter': 100, 'return_all': True, 'disp': False})
                    r = getRMSE(out.x, gt)
                    d = getDiceScore(out.x, gt)
                    conv = out.success

                    dice = np.append(dice, d)
                    rmse = np.append(rmse, r)

                    converge = np.append(converge, conv)

                l2norm = np.linalg.norm(objective_function_gradient_ct(out.x, f, K, Lpred, epsilon))
                max_dice = np.max(dice)
                idx_d = np.where(dice == max_dice)[0]
                idx_r = np.argmin(rmse[idx_d])
                idx = idx_d[idx_r]

                last_rmse = rmse[-1]
                last_dice = dice[-1]
                r_diff = rmse[idx] - last_rmse
                d_diff = dice[idx] - last_dice

                result_idx = np.append(result_idx, idx)
                result_r_diff = np.append(result_r_diff, r_diff)
                result_d_diff = np.append(result_d_diff, d_diff)
                result_rmse = np.append(result_rmse, rmse[-1])
                result_dice = np.append(result_dice, dice[-1])
                result_converged = np.append(result_converged, converge[-1])
                result_gradl2norm = np.append(result_gradl2norm, l2norm)

            print("alpha = ", alpha, ", epsilon = ", epsilon, ", dice = ", np.round(result_dice,3))

            record[idx_param, :] = [sigma, beta, alpha, epsilon, np.mean(result_rmse), np.mean(result_dice),
                                    np.mean(result_gradl2norm), np.std(result_gradl2norm),
                                    np.mean(result_converged), np.mean(result_r_diff), np.mean(result_d_diff),
                                    np.mean(result_idx)]

        full_record = np.concatenate((full_record, record), axis=0)

    full_record = np.delete(full_record, (0), axis=0)
    row_ind_rmse = full_record[:, 4].argmin()
    print("Best combination according to rmse is: ", full_record[row_ind_rmse, :])
    row_ind_dice = full_record[:, 5].argmax()
    print("Best combination according to dice score is: ", full_record[row_ind_dice, :])

    return full_record


if __name__ == '__main__':

    dataset = MNIST('', train=False, download=True)
    digits_data = dataset.data.detach().numpy()
    digits_target = dataset.targets.detach().numpy()
    idx = np.where(digits_target == 0)[0]
    X_train, X_test, y_train, y_test = train_test_split(
        digits_data[idx], digits_target[idx], train_size=800, test_size=100, shuffle=True
    )
    X_learn, X_tune, y_learn, y_tune = train_test_split(
        X_train, y_train, train_size=500, test_size=300, shuffle=True
    )

    X_test_data = np.zeros((X_test.shape[0], 28 ** 2))
    for i in range(X_test.shape[0]):
        X_test_data[i, :] = X_test[i, :].ravel()
    X_tune_data = np.zeros((X_tune.shape[0], 28 ** 2))
    for i in range(X_tune.shape[0]):
        X_tune_data[i, :] = X_tune[i, :].ravel()
    X_learn_data = np.zeros((X_learn.shape[0], 28 ** 2))
    for i in range(X_learn.shape[0]):
        X_learn_data[i, :] = X_learn[i, :].ravel()

    np.savetxt("CT_X_test.csv", X_test_data, delimiter=",")
    np.savetxt("CT_X_tune.csv", X_tune_data, delimiter=",")
    np.savetxt("CT_X_learn.csv", X_learn_data, delimiter=",")


    u_tune_data = get_binary_images(X_tune, 128)
    u_tune = np.zeros((u_tune_data.shape[0], 28, 28))
    for i in range(u_tune_data.shape[0]):
        u_tune[i, :] = u_tune_data[i, :].reshape(28,28)


    angles = np.linspace(0, 2*np.pi, 72, False)
    [K_5, f_03_5] = sinograph_generator(u_tune, 0.3, angles)
    [K_5, f_07_5] = sinograph_generator(u_tune, 0.7, angles)
    [K_5, f_10_5] = sinograph_generator(u_tune, 1, angles)

    angles = np.linspace(0, 2*np.pi, 36, False)
    [K_10, f_03_10] = sinograph_generator(u_tune, 0.3, angles)
    [K_10, f_07_10] = sinograph_generator(u_tune, 0.7, angles)
    [K_10, f_10_10] = sinograph_generator(u_tune, 1, angles)

    angles = np.linspace(0, 2 * np.pi, 18, False)
    [K_20, f_03_20] = sinograph_generator(u_tune, 0.3, angles)
    [K_20, f_07_20] = sinograph_generator(u_tune, 0.7, angles)
    [K_20, f_10_20] = sinograph_generator(u_tune, 1, angles)

    L4 = getLaplacian_Sparse(28)
    alpha_range5 = [0.0004, 0.0001, 0.00005]
    alpha_range10 = [0.0008, 0.0004, 0.0001]
    alpha_range20 = [0.001, 0.0005, 0.0001]

    epsilon_range = [0.1, 0.05, 0.01, 0.005]
    beta_range = [5, 10, 15]

    K_list = [K_5, K_5, K_5, K_10, K_10, K_10, K_20, K_20, K_20]
    f_list = [f_03_5, f_07_5, f_10_5, f_03_10, f_07_10, f_10_10, f_03_20, f_07_20, f_10_20]
    sigma_list = [0.3, 0.7, 1, 0.3, 0.7, 1, 0.3, 0.7, 1]
    angle_list = [5, 5, 5, 10, 10, 10, 20, 20, 20]
    alpha_list = [alpha_range5, alpha_range5, alpha_range5, alpha_range10,
                  alpha_range10, alpha_range10, alpha_range20, alpha_range20, alpha_range20]

    for angle, sigma, K, f_delta, alpha_range in zip(angle_list, sigma_list, K_list, f_list, alpha_list):
        out_L4 = tune_ct_rec(np.zeros(28 ** 2), K, f_delta[0:40], L4, sigma, alpha_range, epsilon_range, u_tune,
                                num_iter=1500)
        output_str_L4 = 'CT_L4_result_sigma' + str(sigma) + '_angle' + str(angle) + ".csv"
        np.savetxt(output_str_L4, out_L4, delimiter=",")

        out_Lpred = tune_ct_rec_Lpred(np.zeros(28 ** 2), K, f_delta[0:40], beta_range, alpha_range, epsilon_range,
                                      u_tune, num_iter=1500)
        output_str_Lpred = 'CT_Lpred_result_sigma' + str(sigma) + '_angle' + str(angle) + ".csv"
        np.savetxt(output_str_Lpred, out_Lpred, delimiter=",")


###############################Learn Lpred###########################################
        # u_learn_image = get_binary_images(X_learn, 128)
        # u_learnL = np.zeros((u_learn_image.shape[0], 28 ** 2))
        # for i in range(u_learn_image.shape[0]):
        #     u_learnL[i, :] = u_learn_image[i, :].ravel()
        # [Lpred, Wpred, err, gl3sr] = fit_laplacian(u_learnL.astype(float), beta, 0.001, maxiter=100)
        # np.savetxt("Lpred.csv", Lpred, delimiter=",")