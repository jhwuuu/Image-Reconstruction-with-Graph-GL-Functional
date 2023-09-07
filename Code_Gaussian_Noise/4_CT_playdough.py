from basic_functions import *

import itertools
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.data import shepp_logan_phantom, binary_blobs
from skimage.draw import disk
from scipy import sparse
import scipy
from datetime import datetime
import sympy as sp
from matplotlib import colors

import GL_3SR
from numpy import genfromtxt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable


from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from flextomo import projector  # Reconstruct
from flexcalc import process
import astra



def objective_function_ct(u, f_delta, K, L,  epsilon):

    out = 1 / 2 * ((K@u.ravel() - f_delta.ravel()) ** 2).sum() + 1 / 2 * u.ravel().transpose() @ L @ u.ravel() + 1 / epsilon * DoubleWell(u.ravel()).sum()

    return out


def objective_function_gradient_ct(u, f_delta, K, L,  epsilon):

    out =  np.transpose(K) @ ( K@u.ravel() - f_delta.ravel()) + L @ u.ravel() + 1 / epsilon * getWGradient(u.ravel())

    return out


def get_intermediate_objective_value_ct(inter_x, f_delta, K, L, epsilon):
  obj_value = np.array([])
  inter_x = np.array(inter_x)
  for i in range(inter_x.shape[0]):
    obj_value = np.append(obj_value, np.sum(np.abs(objective_function_ct(inter_x[i], f_delta, K, L, epsilon))))
  return obj_value


def get_intermediate_objective_gradient_ct(inter_x, f_delta, K, L,  epsilon):
  obj_gradient_value = np.array([])
  inter_x = np.array(inter_x)
  for i in range(inter_x.shape[0]):
    obj_gradient_value = np.append(obj_gradient_value, np.linalg.norm(objective_function_gradient_ct(inter_x[i], f_delta, K, L,  epsilon)))
  return obj_gradient_value

# alpha is stepsize
def pgd_ct(nx, K, f_delta, L, num_iter=2000, alpha=0.001, epsilon=0.05):

    u = np.zeros(nx**2)
    v = u - alpha * K.transpose() @ ( K@u - f_delta.ravel() )
    out = scipy.optimize.minimize(objective_function, np.zeros(nx**2), (v, L, alpha, epsilon), method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                          options={'maxiter':200, 'return_all':True, 'disp':False})
    all_vecs_L = np.zeros((num_iter, nx**2))
    for iter in range(num_iter):
        u = out.x
        v = u - alpha * K.transpose() @ ( K@u - f_delta.ravel() )
        out = scipy.optimize.minimize(objective_function, np.zeros(nx**2), (v, L, alpha, epsilon), method='Newton-CG', jac=objective_function_gradient, tol=1e-06,
                                              options={'maxiter':200, 'return_all':True, 'disp':False})

        all_vecs_L[iter, :] = out.x

    return all_vecs_L



if __name__ == '__main__':

    # ## determine the scaling factor for reconstruction such that GL regularisation will work better
    # ## (used FBP_CUDA on full data)
    # ## only need to run this piece once

    # path = 'Code/LetterO/Sn0.2Cu0.25Al0.5'
    # proj, geom = process.process_flex(path, correct='cwi-flexray-2023-08-21')
    #
    # [num_slices, num_angles, det_count] = proj.shape
    # rec_im_size = det_count
    # array = [0, 1, 3, 4, 6, 7]
    # ## definition: astra_create_proj_geom('fanflat_vec', det_count, vectors)
    # proj_geom = astra.create_proj_geom('fanflat_vec', det_count, geom.get_vectors(num_angles)[:, array])
    # vol_geom = geom.astra_volume_geom([1, proj.shape[2], proj.shape[2]])
    # del vol_geom['option']['WindowMinZ']
    # del vol_geom['option']['WindowMaxZ']
    # del vol_geom['GridSliceCount']
    #
    # proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)
    # f_delta = proj[proj.shape[0] // 2, :, :]
    # f_delta_id = astra.data2d.create('-sino', proj_geom, data=f_delta)
    # rec_id = astra.data2d.create('-vol', vol_geom)
    #
    # cfg = astra.astra_dict('FBP_CUDA')
    # cfg['ReconstructionDataId'] = rec_id
    # cfg['ProjectorId'] = proj_id
    # cfg['ProjectionDataId'] = f_delta_id
    # ## Create the algorithm object from the configuration structure
    # alg_id = astra.algorithm.create(cfg)
    # ## Run the algorithm
    # astra.algorithm.run(alg_id)
    # ## Get the result
    # u_rec_fbp = astra.data2d.get(rec_id)
    #
    # ## scaling factor decided by OTSU
    # thresh = threshold_otsu(u_rec_fbp)
    # u_binary = u_rec_fbp > thresh
    # print(thresh)
    # scaling = np.mean(u_rec_fbp[u_binary == 1])
    # print(scaling)

    ## reconstruction using graph GL regularisation
    ## subsample 360 projections
    path = 'Code/LetterO/Sn0.2Cu0.25Al0.5'
    binning = 8
    skip = 5
    proj, geom = process.process_flex(path, sample=binning, skip=skip, correct='cwi-flexray-2023-08-21')
    [num_slices, num_angles, det_count] = proj.shape
    ## Recreate the projection geometry using Astra
    rec_im_size = det_count
    array = [0,1,3,4,6,7]
    ## definition: astra_create_proj_geom('fanflat_vec', det_count, vectors)
    proj_geom = astra.create_proj_geom('fanflat_vec', det_count, geom.get_vectors(num_angles)[:, array])
    vol_geom = geom.astra_volume_geom([1, proj.shape[2], proj.shape[2]])
    del vol_geom['option']['WindowMinZ']
    del vol_geom['option']['WindowMaxZ']
    del vol_geom['GridSliceCount']

    proj_id = astra.create_projector('line_fanflat', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)
    # Get the projection matrix as a Scipy sparse matrix.
    K = astra.matrix.get(matrix_id)
    ## decide alpha range
    # print(2/np.linalg.norm((np.transpose(K)@K).toarray()))

    ## scaling factor = 0.025758242 using FBP_CUDA with full data
    scaling = 0.025758242
    ## recon with own method
    L4 = getLaplacian_Sparse(rec_im_size)
    ## extract the sinogram of the central slice
    sino = proj[proj.shape[0] // 2, :, :]

    alpha = 5e-05
    epsilon = 0.03
    all_vecs_L4 = pgd_ct(rec_im_size, K, sino/scaling, L4, num_iter=9000, alpha=alpha, epsilon=epsilon)

    obj_scipy_L4 = get_intermediate_objective_value_ct(all_vecs_L4, sino/scaling, K, L4, epsilon)
    obj_grad_scipy_L4 = get_intermediate_objective_gradient_ct(all_vecs_L4, sino/scaling, K, L4, epsilon)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(np.log(obj_scipy_L4), label='L4', color='royalblue')
    ax[1].plot(np.log(obj_grad_scipy_L4), label='L4', color='royalblue')
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel('Log objective value')
    ax[1].set_xlabel('Number of iterations')
    ax[1].set_ylabel('Log-norm of objective function gradient')
    plt.tight_layout()
    plt.show()

    u_rec_L4 = all_vecs_L4[-1,:]
    u_rec_L4_b = np.where(u_rec_L4>0.5,1,0)
    print(u_rec_L4.max())
    print(u_rec_L4.min())
    fig, ax = plt.subplots(1, 2, layout='compressed')
    img = ax[0].imshow(u_rec_L4.reshape(rec_im_size,rec_im_size), cmap='plasma')
    ax[1].imshow(u_rec_L4_b.reshape(rec_im_size,rec_im_size), cmap='plasma')
    cbar2 = fig.colorbar(img, ax=ax, location='right', format='%.4f')
    plt.show()

