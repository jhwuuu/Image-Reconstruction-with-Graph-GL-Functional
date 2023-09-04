import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom, binary_blobs
from skimage.draw import disk
from scipy import sparse
import scipy
from datetime import datetime
import sympy as sp


#################### generate phantom ####################

def getPhantom(nx, r=0.9):
    """
    (Note: This function is copied from xCTing_training. The output is actually a 2d array)

    Define phantom image.

    input:
        nx - dimension of the input image
        r - radius of phantom, optional (default = 0.9)

    output:
        u - phantom image as 1d array of length nx*nx
    """
    # mask
    mask = np.zeros((nx, nx))
    #### disk: generate grid data, disk(center, radius, shape)
    ii, jj = disk((nx // 2, nx // 2), r * (nx // 2))
    mask[ii, jj] = 1

    # binary blobs
    #### binary_blobs: in the Scikit package, blob_size_fraction: Typical linear size of blob, as a fraction of length
    u = np.float64(binary_blobs(length=nx, blob_size_fraction=0.5))
    u *= mask

    # return
    return u


def getNoisyImage(u, sigma):
    """
    Generate noisy image. Noise are uniformly distributed.

    input:
        u - ground truth image
        sigma - noise level

    output:
        v - noise image as 2d array of shape (nx,nx)
    """

    return u + sigma * np.random.rand(u.shape[0], u.shape[1])



def getNoisyImage_Gaussian(u, sigma):
    """
    Generate noisy image. Noise are uniformly distributed.

    input:
        u - ground truth image
        sigma - noise level

    output:
        v - noise image as 2d array of shape (nx,nx)
    """
    # return u + sigma * np.random.rand(u.shape[0], u.shape[1])
    return u + np.random.default_rng().normal(0, sigma, (u.shape[0],u.shape[1]))


#################### objective function ####################

def getAdjacency_Sparse(nx):
    """
    Define adjacency matrix (sparse) of the image as a graph.

    input:
        nx - x-axis dimension of the input image

    output:
        A - adjacency matrix of dimension nx^2 * nx^2
    """

    num_pixels = nx ** 2

    # two types of indices
    row_index1 = np.arange(0, num_pixels - 1)
    col_index1 = np.arange(1, num_pixels)

    row_index2 = np.arange(0, num_pixels - nx)
    col_index2 = np.arange(nx, num_pixels)

    # weights
    weight1 = np.ones(row_index1.shape[0])
    weight2 = np.ones(row_index2.shape[0])
    for r in row_index1:
        if (r + 1) % nx == 0:
            weight1[r] = 0

    # create sparse matrix
    rows = np.concatenate((row_index1, row_index2, col_index1, col_index2))
    cols = np.concatenate((col_index1, col_index2, row_index1, row_index2))
    weights = np.concatenate((weight1, weight2, weight1, weight2))

    sparse_matrix = sparse.csc_matrix((weights, (rows, cols)), shape=(num_pixels, num_pixels))

    return sparse_matrix


def getLaplacian_Sparse(nx):
    """
    Define Laplacian matrix (sparse).

    input:
        nx - x-axis dimension of the input image

    output:
        L - Laplacian matrix of dimension nx^2 * nx^2
    """

    Adj = getAdjacency_Sparse(nx)
    deg = sparse.diags(Adj.sum(axis=1).A1)
    Lap = deg - Adj

    return Lap


def DoubleWell(u):
    """
    Define the double well function W(x)=x^2*(x-1)^2.

    input:
        u - the input image as 1d array

    output:
        W(u) - double well function as 1d array
    """

    return u ** 2 * (u - 1) ** 2


def getWGradient(u):
    """
    Define gradient of the double well function W(x)=x^2*(x-1)^2, W'(x)=2x(2x-1)(x-1).

    input:
        u - the input image as 1d array

    output:
        W'(u) - gradient of the double well function as 1d array
    """

    return 2 * u * (2 * u - 1) * (u - 1)


def getWHessian(u):
    """
    Define Hessian of the double well function W(x)=x^2*(x-1)^2, W''(x)=2+12x(x-1).

    input:
        u - the input image as 1d array

    output:
        W''(u) - Hessian of the double well function as diagonal matrix
    """

    out = 2 + 12 * u * (u - 1)
    out = sparse.diags(out.ravel())

    return out


def objective_function(u, v, L, alpha, epsilon):
    """
    Define the objective function f(u).

    input:
        u - the input image as 2d array
        v - the noisy image as 2d array
        L - the Laplacian matrix (sparse)
        alpha - hyperparameter
        epsilon - hyperparameter

    output:
        f(u) - value of the objective function
    """

    out = (1 / (2 * alpha)) * ((
                                           u.ravel() - v.ravel()) ** 2).sum() + 1 / 2 * u.ravel().transpose() @ L @ u.ravel() + 1 / epsilon * DoubleWell(
        u.ravel()).sum()

    return out


def objective_function_gradient(u, v, L, alpha, epsilon):
    """
    Define the gradient of the objective function, f'(u).

    input:
        u - the input image as 2d (or 1d) array
        v - the noisy image as 2d (or 1d) array
        L - the Laplacian matrix (sparse)
        alpha - hyperparameter
        epsilon - hyperparameter

    output:
        f'(u) - value of the gradient of the objective function
    """

    out = 1 / alpha * (u.ravel() - v.ravel()) + L @ u.ravel() + 1 / epsilon * getWGradient(u.ravel())

    return out


def objective_function_Hessian(u, v, L, alpha, epsilon):
    """
    Define the second derivative of the objective function, f''(u).

    input:
        u - the input image as *1d* array
        v - the noisy image as 2d (or 1d) array (Note: this parameter is not used in the function)
        L - the Laplacian matrix (sparse)
        alpha - hyperparameter
        epsilon - hyperparameter

    output:
        f''(u) - Hessian matrix of the objective function
    """

    nx = u.shape[0]
    out = 1 / alpha * sparse.eye(nx) + L + 1 / epsilon * getWHessian(u)

    return out


def getRMSE(u, v):
    """
    Calculate the root mean squared error between two images.

    input:
        u - the ground truth image as *1d* array
        v - the noisy image as *1d* array

    output:
        r - root mean squared error between u and v
    """
    return np.sqrt(np.square(u - v).mean())


def getDiceScore(denoised, ground_truth):
    """
    Calculate the root mean squared error between two images.

    input:
        u - the ground truth image as *1d* array
        v - the noisy image as *1d* array

    output:
        Dice - Dice score between u and v
    """
    # result = denoised.round()   # need 'denoised' values less than 1.5
    result = np.where(denoised > 0.5, 1, 0)
    out = np.sum(result == ground_truth) / denoised.shape[0]

    return out


def is_matrix_psd(m):
    """
    Dicide if the input matrix is positive definite.

    input:
        m - symmetric matrix

    output:
        True/False - True if the input matrix is pd.
    """
    # using 'SA' because 'SM' takes forever
    E, V = scipy.sparse.linalg.eigsh(m, which='SA', k=2)
    print("The two smallest eigenvalues are: ", E)
    if E[0] >= 0:
        return True
    else:
        return False


#################### reduced objective function ####################

def objective_onlyW(x, alpha, epsilon, v):
    """
    Define reduced objective function g(u) = fidelity term + double well potential
    input:
        x - input image as 1d array
        alpha - hyperparameter
        epsilon - hyperparameter
        v - noisy image as 2d (or 1d) array

    output:
        g(u) - reduced objective function value

    """
    return 1 / (2 * alpha) * ((x - v.ravel()) ** 2).sum() + 1 / epsilon * DoubleWell(x.ravel()).sum()


def objective_onlyW_element(x, alpha, epsilon, v):
    """
    Define an element in the reduced objective function g(u)
    input:
        x - scalar
        alpha - hyperparameter
        epsilon - hyperparameter
        v - a pixel value of the noisy image

    output:
        g(u)_i - element of the reduced objective function value

    """
    return 1 / (2 * alpha) * ((x - v) ** 2) + 1 / epsilon * DoubleWell(x)


def objective_onlyW_1stderiv(x, alpha, epsilon, v):
    """
    Define the gradient of the reduced objective function g(u), g'(u)
    input:
        x - input image as 1d array
        alpha - hyperparameter
        epsilon - hyperparameter
        v - noisy image as 2d (or 1d) array

    output:
        g'(u) - gradient of the reduced objective function

    """
    return 1 / epsilon * (4 * x ** 3 - 6 * x ** 2) + (2 / epsilon + 1 / alpha) * x - 1 / alpha * v.ravel()


def objective_onlyW_1stderiv_element(x, alpha, epsilon, v):
    """
    Define an element in the gradient of the reduced objective function g(u)
    input:
        x - scalar
        alpha - hyperparameter
        epsilon - hyperparameter
        v - a pixel value of the noisy image

    output:
        g'(u)_i - element of the reduced objective function value

    """
    return 1 / epsilon * (4 * x ** 3 - 6 * x ** 2) + (2 / epsilon + 1 / alpha) * x - 1 / alpha * v


def get_uhat_symbolic(alpha, epsilon, v):
    """
    Calculate symbolically all 3 roots in each dimension of g'(u)=0.
    input:
        alpha - hyperparameter
        epsilon - hyperparameter
        v - noisy image as 2d (or 1d) array

    output:
        u_hat - roots of g'(u)=0 as an array of shape (nx^2, 3)

    """
    u_hat = np.array([])
    x = sp.symbols('x')
    len = v.ravel().shape[0]
    for i in range(len):
        out = sp.solvers.solve(
            1 / epsilon * (4 * x ** 3 - 6 * x ** 2 + 2 * x) + 1 / alpha * x - 1 / alpha * v.ravel()[i], x)
        # print("Iteration ", i, ": ", out)
        u_hat = np.append(u_hat, out)

    u_hat = u_hat.reshape(-1, 3)

    return u_hat


def get_globalMin(u_hat, alpha, epsilon, v):
    """
    Find the global minimial from 3 roots in each dimension of g'(u)=0.
    input:
        u_hat - roots of g'(u)=0 as an array of shape (nx^2, 3)
        alpha - hyperparameter
        epsilon - hyperparameter
        v - noisy image as 2d (or 1d) array

    output:
        out - global min of g(u) as an 1d array

    """
    out = np.array([])
    for i in range(u_hat.shape[0]):
        if sp.im(u_hat[i][0]) == 0:
            out = np.append(out, u_hat[i][0])
        else:
            f_value1 = objective_onlyW_element(sp.re(u_hat[i][0]), alpha, epsilon, v.ravel()[i])
            f_value2 = objective_onlyW_element(sp.re(u_hat[i][2]), alpha, epsilon, v.ravel()[i])
            if f_value1 < f_value2:
                out = np.append(out, sp.re(u_hat[i][0]))
            else:
                out = np.append(out, sp.re(u_hat[i][2]))

    out = out.astype(np.float64)
    return out


def get_uhat_numeric(alpha, epsilon, v):
    """
    Calculate numerically a root in each dimension of g'(u)=0.
    Note: fsovlve can only find one root, which is influenced by the initial value.
        This function returns the 1st column of the solution returned by symbolic computing (i.e. always the real roots)
    input:
        alpha - hyperparameter
        epsilon - hyperparameter
        v - noisy image as 2d (or 1d) array

    output:
        u_hat_numeric - roots of g'(u)=0 as an array of shape (nx^2, 1)

    """
    u_hat_numeric = np.array([])
    len = v.ravel().shape[0]
    for i in range(len):
        out = scipy.optimize.fsolve(objective_onlyW_1stderiv, 0, args=(alpha, epsilon, v.ravel()[i]), full_output=1)
        if out[2] != 1:
            out = scipy.optimize.fsolve(objective_onlyW_1stderiv, 1, args=(alpha, epsilon, v.ravel()[i]), full_output=1)
        u_hat_numeric = np.append(u_hat_numeric, out[0])

    return u_hat_numeric


#################### hyperparameter tuning with scipy minimiser ####################
####### tune hyperparameters alpha and epsilon with fixed noise level sigma #######

def tune_params_scipy(objective, Jacobian, x0, method_str, sigma, epsilon_range, alpha_range, u, v, L):
    """
    Tune hyperparameters alpha and epsilon with specified initial vector and fixed sigma using Scipy solver.

    input:
        objective - objective function
        Jacobian - gradient of objective function
        x0 - initial vector
        method_str - type of solver, usually Newton-CG
        sigma - noise level
        epsilon_range - values of epsilon to be tuned
        alpha_range - values of alpha to be tuned
        u - ground truth image
        v - noisy image
        L - Laplacian matrix

    output:
        out_x - solutions corresponds to each combination of parameters
        out_result - recordings of the performance of each combination of parameters

    """
    out_x = np.array([])
    out_result = np.array([])

    original_r = getRMSE(u.ravel(), v.ravel())

    for alpha in alpha_range:
        for epsilon in epsilon_range:
            alpha, epsilon = np.round((alpha, epsilon), 3)
            print('Parameters: sigma =', sigma, ', alpha = ', alpha, ', epsilon = ', epsilon, ', original RMSE = ',
                  original_r)

            args = (v, L, alpha, epsilon)
            start = datetime.now()
            result = scipy.optimize.minimize(objective, x0, args, method=method_str, jac=Jacobian, tol=1e-06,
                                             options={'maxiter': 200})
            end = datetime.now()
            print("Tuning this image using zero vector used time: {}".format(end - start), end="\n")
            print('Optimizer exited successfully: ', result.message)
            print('Obj value is: ', result.fun, '. Ground truth obj value is ', objective_function(u,v,L,alpha,epsilon))

            A = objective_function_Hessian(result.x, v, L, alpha, epsilon)
            print(is_matrix_psd(A))

            out_x = np.hstack((out_x, sigma, alpha, epsilon, np.array(result.x)))

            r = getRMSE(u.ravel(), result.x)
            dice = getDiceScore(result.x, u.ravel())
            out_result = np.hstack((out_result, sigma, alpha, epsilon, r, dice, result.success, end - start))
            print('Final RMSE = ', r)
            print('Final Dice score = ', dice)

    out_x = out_x.reshape(len(epsilon_range) * len(alpha_range), -1)
    out_result = out_result.reshape(-1, 7)

    return out_x, out_result


def tune_params_scipy_diffInitial(objective, Jacobian, method_str, sigma, epsilon_range, alpha_range, u, v, L):
    """
    Tune hyperparameters alpha and epsilon with v-dependent initials and fixed sigma using Scipy solver.

    input:
        objective - objective function
        Jacobian - gradient of objective function
        method_str - type of solver, usually Newton-CG
        sigma - noise level
        epsilon_range - values of epsilon to be tuned
        alpha_range - values of alpha to be tuned
        u - ground truth image
        v - noisy image
        L - Laplacian matrix

    output:
        out_x - solutions corresponds to each combination of parameters
        out_result - recordings of the performance of each combination of parameters

    """
    out_x = np.array([])
    out_result = np.array([])

    original_r = getRMSE(u.ravel(), v.ravel())
    for alpha in alpha_range:
        for epsilon in epsilon_range:
            alpha, epsilon = np.round((alpha, epsilon), 3)
            print('Parameters: sigma =', sigma, ', alpha = ', alpha, ', epsilon = ', epsilon, ', original RMSE = ',
                  original_r)

            args = (v, L, alpha, epsilon)

            x0 = get_uhat_numeric(alpha, epsilon, v)

            start = datetime.now()
            result = scipy.optimize.minimize(objective, x0, args, method=method_str, jac=Jacobian, tol=1e-06,
                                             options={'maxiter': 200})
            end = datetime.now()
            print("Tuning this image using local min used time: {}".format(end - start), end="\n")

            print('Optimizer exited successfully: ', result.message)

            out_x = np.hstack((out_x, sigma, alpha, epsilon, np.array(result.x)))

            r = getRMSE(u.ravel(), result.x)
            dice = getDiceScore(result.x, u.ravel())
            out_result = np.hstack((out_result, sigma, alpha, epsilon, r, dice, result.success, end - start))
            print('Final RMSE = ', r)
            print('Final Dice score = ', dice)

    out_x = out_x.reshape(len(epsilon_range) * len(alpha_range), -1)
    out_result = out_result.reshape(-1, 7)

    return out_x, out_result

