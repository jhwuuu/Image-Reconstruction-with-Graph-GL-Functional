from basic_functions import *
import numpy as np
import astra
import pylops
from pylops import LinearOperator


def get_binary_images(data, threshold):

    return np.where(data > threshold, 1, 0)

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

class RandonMat(LinearOperator):
  def __init__(self, d, nrow, ncol, dtype=None):
    self.d = d
    # self.shape = (1512, 784)
    self.shape = (nrow, ncol)
    self.dtype = np.dtype(dtype)
    self.explicit = False
    self.matvec_count = 0
    self.rmatvec_count = 0

  def _matvec(self, x):
    return self.d * x

  def _rmatvec(self, x):
    return self.d.T * x


def tst_ct_TV(K_operator, f_delta_data, ground_truth_data, sigma, mu,
               num_pixels_x=28, damp_lambda=[1.0,1.0], niter=3, niterinner=2):
    num_data = f_delta_data.shape[0]
    nx = num_pixels_x
    Dop = [
      pylops.FirstDerivative(
          (nx, nx), axis=0, edge=True, kind="backward", dtype=np.float64
      ),
      pylops.FirstDerivative(
          (nx, nx), axis=1, edge=True, kind="backward", dtype=np.float64
      ),
    ]
    result_rmse = np.array([])
    result_dice = np.array([])
    result_u_rec = np.zeros((num_data, num_pixels_x**2))
    for idx_data in range(num_data):
      f = f_delta_data[idx_data,:]
      gt = ground_truth_data[idx_data,:].ravel()

      xinv = pylops.optimization.sparsity.splitbregman(
              K_operator,
              f,
              Dop,
              niter_outer=niter,
              niter_inner=niterinner,
              mu=mu,
              epsRL1s=damp_lambda,
              tol=1e-4,
              tau=1.0,
              show=False,
              **dict(iter_lim=20, damp=1e-2)
            )[0]
      out_x = np.real(xinv)
      r = getRMSE(out_x, gt)
      d = getDiceScore(out_x, gt)
      result_rmse = np.append(result_rmse, r)
      result_dice = np.append(result_dice, d)

      result_u_rec[idx_data,:] = out_x

    print("TV: mu = ", mu, ", sigma = ", sigma)
    print("Average RMSE is: ", np.mean(result_rmse), ". Std of RMSE is: ", np.std(result_rmse))
    print("Average Dice score is: ", np.mean(result_dice), ". Std of Dice score is: ", np.std(result_dice))
    print("TV Dice: ", '$', np.round( np.mean(result_dice), 4), '\pm',  np.round(np.std(result_dice), 4), '$')
    print("TV RMSE: ", '$', np.round(np.mean(result_rmse), 4), '\pm', np.round(np.std(result_rmse), 4), '$')
    return result_u_rec


def tst_ct_algebraic(f_delta_data, ground_truth_data, angles, algo='SIRT',
                      num_iter=30, num_pixels_x=28, det_width=1, originToSource=100, originToDetector=50):

    u_rec_data = np.zeros((f_delta_data.shape[0], num_pixels_x**2))
    rmse = np.array([])
    dice = np.array([])

    num_detectors = int(1.5*num_pixels_x)
    num_angles = angles.shape[0]

    vol_geom = astra.create_vol_geom(num_pixels_x, num_pixels_x)
    proj_geom = astra.create_proj_geom('fanflat', det_width, num_detectors, angles, originToSource, originToDetector)
    proj_id = astra.create_projector('strip_fanflat', proj_geom, vol_geom)

    ## Create a data object for the reconstruction
    rec_id = astra.data2d.create('-vol', vol_geom)

    ## Set up the parameters for a reconstruction algorithm
    cfg = astra.astra_dict(algo)
    cfg['ReconstructionDataId'] = rec_id
    cfg['ProjectorId'] = proj_id

    for idx in range(f_delta_data.shape[0]):
        f_delta = f_delta_data[idx,:].reshape((num_angles, num_detectors))
        f_delta_id = astra.data2d.create('-sino', proj_geom, data=f_delta)

        cfg['ProjectionDataId'] = f_delta_id
        ## Create the algorithm object from the configuration structure
        alg_id = astra.algorithm.create(cfg)
        ## Run the algorithm
        astra.algorithm.run(alg_id, num_iter)
        ## Get the result
        u_rec = astra.data2d.get(rec_id)
        u_rec_data[idx,:] = u_rec.ravel()

        r = getRMSE(u_rec.ravel(), ground_truth_data[idx,:])
        d = getDiceScore(u_rec.ravel(), ground_truth_data[idx,:])
        rmse = np.append(rmse, r)
        dice = np.append(dice, d)

    # Clean up
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)
    astra.data2d.delete(f_delta_id)
    astra.projector.delete(proj_id)

    print("SIRT: ")
    print("Average RMSE is: ", np.mean(rmse), ". Std of RMSE is: ", np.std(rmse))
    print("Average Dice score is: ", np.mean(dice), ". Std of Dice score is: ", np.std(dice))
    print("SIRT Dice: ", '$', np.round(np.mean(dice), 4), '\pm', np.round(np.std(dice), 4), '$')
    print("SIRT RMSE: ", '$', np.round(np.mean(rmse), 4), '\pm', np.round(np.std(rmse), 4), '$')
    return u_rec_data


def tst_ct_L(initial, K, f_delta_data, ground_truth_data, L, sigma, alpha, epsilon, num_iter=2000, num_pixels_x=28):
    num_data = f_delta_data.shape[0]
    nx = initial.shape[0]
    result_rmse = np.array([])
    result_dice = np.array([])
    result_u_rec = np.zeros((num_data, num_pixels_x**2))
    for idx_data in range(num_data):

        f = f_delta_data[idx_data,:]
        gt = ground_truth_data[idx_data,:].ravel()
        
        u = initial.copy()
        v = u - alpha * K.transpose() @ ( K@u - f)
        out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, L, alpha, epsilon), method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                              options={'maxiter':100, 'return_all':True, 'disp':False})
        for iter in range(num_iter):
            u = out.x
            v = u - alpha * K.transpose() @ ( K@u - f)
            out = scipy.optimize.minimize(objective_function, np.zeros(nx), (v, L, alpha, epsilon), method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                                  options={'maxiter':100, 'return_all':True, 'disp':False})


        r = getRMSE(out.x, gt)
        d = getDiceScore(out.x, gt)
        result_rmse = np.append(result_rmse, r)
        result_dice = np.append(result_dice, d)
        result_u_rec[idx_data,:] = out.x

    print("alpha = ", alpha, ", epsilon = ", epsilon, ", sigma = ", sigma)
    print("Average RMSE is: ", np.mean(result_rmse), ". Std of RMSE is: ", np.std(result_rmse))
    print("Average Dice score is: ", np.mean(result_dice), ". Std of Dice score is: ", np.std(result_dice))
    print("L Dice: ", '$', np.round(np.mean(result_dice), 4), '\pm', np.round(np.std(result_dice), 4), '$')
    print("L RMSE: ", '$', np.round(np.mean(result_rmse), 4), '\pm', np.round(np.std(result_rmse), 4), '$')

    return result_u_rec

def tst_wrapper(f_test_data, u_test, sigma, between_angle, K, TV_mu, L4_param, Lp_param):

    angles = np.linspace(0, 2 * np.pi, int(360 / between_angle), False)
    out_alg = tst_ct_algebraic(f_test_data, u_test, angles, algo='SIRT', num_iter=25)

    nrow = int(42 * (360/between_angle))
    K_operator = RandonMat(K, nrow, 784)
    out_TV = tst_ct_TV(K_operator, f_test_data, u_test, sigma, TV_mu)


    print("GL with L4: ")
    L4 = getLaplacian_Sparse(28)
    [alpha, epsilon] = L4_param
    out_L4 = tst_ct_L(np.zeros(28 ** 2), K, f_test_data, u_test, L4, sigma, alpha, epsilon, num_iter=2000,
                        num_pixels_x=28)

    print("GL with Lpred: ")
    [beta, alpha, epsilon] = Lp_param
    Lpred_str = "CT_Lpred_"+str(beta)+".csv"
    Lpred = np.genfromtxt(Lpred_str, delimiter=",")
    out_Lpred = tst_ct_L(np.zeros(28 ** 2), K, f_test_data, u_test, Lpred, sigma, alpha, epsilon, num_iter=2000,
                      num_pixels_x=28)


    return out_L4, out_Lpred, out_TV, out_alg

if __name__ == '__main__':

    X_test = np.genfromtxt("CT_X_test.csv", delimiter=",")
    u_test = get_binary_images(X_test, 128)
    u_test_image = np.zeros((u_test.shape[0], 28, 28))
    for i in range(u_test.shape[0]):
        u_test_image[i, :] = u_test[i, :].reshape(28, 28)

    angles = np.linspace(0, 2 * np.pi, 72, False)
    [K_5, f_03_5_test] = sinograph_generator(u_test_image, 0.3, angles)
    [K_5, f_07_5_test] = sinograph_generator(u_test_image, 0.7, angles)
    [K_5, f_10_5_test] = sinograph_generator(u_test_image, 1, angles)

    angles = np.linspace(0, 2 * np.pi, 36, False)
    [K_10, f_03_10_test] = sinograph_generator(u_test_image, 0.3, angles)
    [K_10, f_07_10_test] = sinograph_generator(u_test_image, 0.7, angles)
    [K_10, f_10_10_test] = sinograph_generator(u_test_image, 1, angles)

    angles = np.linspace(0, 2 * np.pi, 18, False)
    [K_20, f_03_20_test] = sinograph_generator(u_test_image, 0.3, angles)
    [K_20, f_07_20_test] = sinograph_generator(u_test_image, 0.7, angles)
    [K_20, f_10_20_test] = sinograph_generator(u_test_image, 1, angles)

    tst_wrapper(f_03_5_test, u_test, 0.3, 5, K_5, 1, [0.0004, 0.005], [10, 0.0001, 0.005])
    tst_wrapper(f_07_5_test, u_test, 0.7, 5, K_5, 1, [0.0004, 0.005], [10, 0.0001, 0.005])
    tst_wrapper(f_10_5_test, u_test, 1, 5, K_5, 0.5, [0.0004, 0.01], [10, 0.0001, 0.01])

    tst_wrapper(f_03_10_test, u_test, 0.3, 10, K_10, 1, [0.0001, 0.01], [10, 0.0008, 0.01])
    tst_wrapper(f_07_10_test, u_test, 0.7, 10, K_10, 1, [0.0008, 0.05], [10, 0.0001, 0.01])
    tst_wrapper(f_10_10_test, u_test, 1, 10, K_10, 0.5, [0.0004, 0.01], [10, 0.0004, 0.01])

    tst_wrapper(f_03_20_test, u_test, 0.3, 20, K_20, 1.5, [0.001, 0.05], [10, 0.0005, 0.05])
    tst_wrapper(f_07_20_test, u_test, 0.7, 20, K_20, 1, [0.001, 0.05], [10, 0.0005, 0.01])
    tst_wrapper(f_10_20_test, u_test, 1, 20, K_20, 0.5, [0.0005, 0.01], [10, 0.0005, 0.01])



