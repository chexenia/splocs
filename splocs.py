import argparse
import os
import numpy as np
import pymesh
from scipy.linalg import svd, norm, cho_factor, cho_solve
import h5py
from inout import load_animation, save_animation, load_splocs, save_splocs
from geodesic import GeodesicDistanceComputation
import time

def project_weight(x):
    x = np.maximum(0., x)
    max_x = x.max()
    if max_x == 0:
        return x
    else:
        return x / max_x

def prox_l1l2(Lambda, x, beta):
    xlen = np.sqrt((x**2).sum(axis=-1))
    with np.errstate(divide='ignore'):
        shrinkage = np.maximum(0.0, 1 - beta * Lambda / xlen)
    return (x * shrinkage[...,np.newaxis])

def compute_support_map(idx, geodesics, min_dist, max_dist):
    phi = geodesics(idx)
    return (np.clip(phi, min_dist, max_dist) - min_dist) / (max_dist - min_dist)

def compute_splocs(input_animation_file, output_sploc_file, output_animation_file, args):
    rest_shape = args.reference_shape
    K =  args.ncomponents
    smooth_min_dist = args.smooth_min_dist
    smooth_max_dist = args.smooth_max_dist
    num_iters_max = args.max_iters
    sparsity_lambda = args.sparsity
    rho = args.rho
    num_admm_iterations = args.num_admm_iters
    save_num_iters = args.save_each_iters

    start_time = time.time()

    verts, tris, _, vmean, vscale = load_animation(input_animation_file)

    F, N, _ = verts.shape

    if rest_shape is None:
        Xmean = np.mean(verts, axis=0)
    else:
        Xmean = pymesh.load_mesh(rest_shape).vertices

    print('saving mean to animation_mean.obj')
    mesh = pymesh.form_mesh(vertices=Xmean, faces=tris)
    pymesh.save_mesh('animation_mean.obj', mesh)

    # prepare geodesic distance computation on the restpose mesh
    compute_geodesic_distance = GeodesicDistanceComputation(Xmean, tris)
    # form animation matrix, subtract mean and normalize
    # notice that in contrast to the paper, X is an array of shape (F, N, 3) here
    X = verts - Xmean[np.newaxis]
    pre_scale_factor = 1 / np.std(X)
    X *= pre_scale_factor
    R = X.copy() # residual

    C = []
    W = []
    for k in range(K):
        # find the vertex explaining the most variance across the residual animation
        magnitude = (R**2).sum(axis=2)
        idx = np.argmax(magnitude.sum(axis=0))
        #idx = C_index[k]
        # find linear component explaining the motion of this vertex

        U, s, Vt = svd(R[:,idx,:].reshape(R.shape[0], -1).T, full_matrices=False)
        wk = s[0] * Vt[0,:] # weights
        # invert weight according to their projection onto the constraint set
        # this fixes problems with negative weights and non-negativity constraints
        wk_proj = project_weight(wk)
        wk_proj_negative = project_weight(-wk)
        wk = wk_proj \
                if norm(wk_proj) > norm(wk_proj_negative) \
                else wk_proj_negative
        s = 1 - compute_support_map(idx, compute_geodesic_distance, smooth_min_dist, smooth_max_dist)
        # solve for optimal component inside support map
        ck = (np.tensordot(wk, R, (0, 0)) * s[:,np.newaxis])\
                / np.inner(wk, wk)

        C.append(ck)
        W.append(wk)
        # update residual
        R -= np.outer(wk, ck).reshape(R.shape)
    C = np.array(C)
    W = np.array(W).T

    # prepare auxiluary variables
    Lambda = np.empty((K, N))
    U = np.zeros((K, N, 3))

    # main global optimization
    for it in range(num_iters_max):
        it_time = time.time()

        # update weights
        Rflat = R.reshape(F, N*3) # flattened residual
        for k in range(C.shape[0]): # for each component
            Ck = C[k].ravel()
            Ck_norm = np.inner(Ck, Ck)
            if Ck_norm <= 1.e-8:
                # the component seems to be zero everywhere, so set it's activation to 0 also
                W[:,k] = 0
                continue # prevent divide by zero
            # block coordinate descent update
            Rflat += np.outer(W[:,k], Ck)
            opt = np.dot(Rflat, Ck) / Ck_norm
            W[:,k] = project_weight(opt)
            Rflat -= np.outer(W[:,k], Ck)
        # update spatially varying regularization strength

        for k in range(K):
            ck = C[k]
            # find vertex with biggest displacement in component and compute support map around it
            idx = (ck**2).sum(axis=1).argmax()
            support_map = compute_support_map(idx, compute_geodesic_distance,
                                              smooth_min_dist, smooth_max_dist)
            # update L1 regularization strength according to this support map
            Lambda[k] = sparsity_lambda * support_map
        # update components
        Z = C.copy() # dual variable
        # prefactor linear solve in ADMM
        G = np.dot(W.T, W)
        c = np.dot(W.T, X.reshape(X.shape[0], -1))
        solve_prefactored = cho_factor(G + rho * np.eye(G.shape[0]))
        # ADMM iterations
        for admm_it in range(num_admm_iterations):
            C = cho_solve(solve_prefactored, c + rho * (Z - U).reshape(c.shape)).reshape(C.shape)
            Z = prox_l1l2(Lambda, C + U, 1. / rho)
            U = U + C - Z
        # set updated components to dual Z,
        # this was also suggested in [Boyd et al.] for optimization of sparsity-inducing norms
        C = Z
        # evaluate objective function
        R = X - np.tensordot(W, C, (1, 0)) # residual
        sparsity = np.sum(Lambda * np.sqrt((C**2).sum(axis=2)))
        e = (R**2).sum() + sparsity
        
        # TODO convergence check
        print("iteration %03d, E=%f, time:%fs" % (it, e, time.time() - it_time))

        if it % save_num_iters == 0:
            save_splocs(os.path.splitext(output_sploc_file)[0]+str(it)+'.h5', Xmean, tris, C / pre_scale_factor, vmean, vscale)

    # undo scaling
    C /= pre_scale_factor

    # save resulting components
    save_splocs(output_sploc_file, Xmean, tris, C, vmean, vscale)

    if output_animation_file is not None:
        # save encoded animation including the weights
        vs = np.tensordot(W, C, (1, 0)) + Xmean[np.newaxis]
        save_animation(output_animation_file, vs, tris, W)

    print("total time elapsed", time.time() - start_time)

def fit_weights(input_animation_file, input_sploc_file, output_reconstructed_data):
    """@input_animation_file is the aligned hdf5 for the test data
       @input_sploc_file is the already processed splocs components
       @output_reconstructed_data is the file to save the reconstructed components and their weights
    """
    verts, tris, _, _, _ = load_animation(input_animation_file)

    print("sample shape", verts.shape, tris.shape)     
    Xmean, tris_splocs, C, names, indices, vmean, vscale = load_splocs(input_sploc_file)
    assert(tris.shape == tris_splocs.shape, "animation should have same topology")
    assert(np.all(tris == tris_splocs), "animation should have same topology")

    # fit weights
    X = (verts - Xmean)
    # pre-invert component subspace so that all frames can be optimized very efficiently
    Cflat = C.reshape(C.shape[0], -1)
    Cinv = np.linalg.pinv(np.dot(Cflat, Cflat.T) + np.eye(Cflat.shape[0]))
    
    # optimize for weights of each frame
    W = np.array(
        [np.dot(Cinv, np.dot(Cflat, x.ravel())) for x in X])
    # reconstruct animation
    verts_reconstructed = np.tensordot(W, C, (1, 0)) + Xmean[np.newaxis]
    #[save_obj(verts_reconstructed[i, :, :] * 200, tris, 'reconstructed'+str(i)+'.obj') for i in range(10, 20, 2)]
    
    save_animation(output_reconstructed_data, verts_reconstructed, tris, W)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Find Sparse Localized Deformation Components')
    parser.add_argument('operation', help='specify train or test')
    parser.add_argument('input_animation_file', help='prepare it first with inout.py convert_sequence_to_hdf5')
    parser.add_argument('--splocs-file', help='path to ouput splocs for train or input for test')
    parser.add_argument('--output-animation', default=None, help='output animation file (will also save the component weights)')
    parser.add_argument('--reference-shape', default=None, help='which frame to use as reference shape, use mean as default or provide a path to load mesh')
    parser.add_argument('--ncomponents', type=int, default=50, help='number of components to calculate')
    parser.add_argument('--smooth-min-dist', default=0.1, help='minimum geodesic distance for support map, d_min_in paper')
    parser.add_argument('--smooth-max-dist', default=0.4, help='maximum geodesic distance for support map, d_min_in paper')
    parser.add_argument('--max-iters', type=int, default=10, help='number of iterations to run')
    parser.add_argument('--sparsity', default=2.0, help='sparsity parameter, lambda in paper')
    parser.add_argument('--rho', default = 10, help='penalty parameter for ADMM')
    parser.add_argument('--num-admm-iters', type=int, default = 10, help='number of ADMM iterations')
    parser.add_argument('--save-each-iters', type=int, default = 10, help='save each number of iterations')

    args = parser.parse_args()

    if args.operation == 'train':
        compute_splocs(args.input_animation_file,
                        args.splocs_file,
                        args.output_animation, args)
    elif args.operation == 'test':
        fit_weights(args.input_animation_file, args.splocs_file, args.output_animation)
    else:
        raise ValueError('unsupported operation', args.operation)

