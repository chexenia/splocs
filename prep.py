import os
import numpy as np
import h5py
import argparse
import reco2.utilities.utils3d as u3d
from inout import load_splocs
import reco2.utilities.h5data as h5d
import pymesh

def convert(src, dst, filepattern):
    """Read splocs from hdf5 files and convert in to paraformer compatible"""
    Xmean,  tris, C, names, indices, vmean, vscale = load_splocs(src)
    mean = (Xmean * vscale + vmean)
    #save mean
    mean_mesh = pymesh.form_mesh(mean, tris)
    if not os.path.exists(dst):
        os.makedirs(dst)
    pymesh.save_mesh(os.path.join(dst, 'splocs_mean' + filepattern), mean_mesh)

    pm = h5d.H5PCA(os.path.join(dst, 'splocs.h5'))

    ncomps = C.shape[0]
    #again we have row-column incompatibility between cpp eigen representation and python
    Ct = C.reshape((C.shape[0], -1)).T.reshape(C.shape[0], -1)
    print(Ct[0], Ct.shape)
    comps = Ct#Ct.reshape((Ct.shape[1], Ct.shape[0])).T
    print(mean.shape, comps.shape)
    pm._set(mean, comps, np.zeros(ncomps), 0)
    pm.close()

    #generate test sample and save
    W = np.random.rand(ncomps)
    sample = np.tensordot(W, C, (0, 0)) + Xmean#np.dot(W.T, Ct) + Xmean.flatten()#
    sample_mesh = pymesh.form_mesh(sample.reshape(Xmean.shape), tris)
    pymesh.save_mesh(os.path.join(dst, 'sample' + filepattern), sample_mesh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('src', help='path to splocs file')
    parser.add_argument('dst', help='outpath to h5pca file and mean.obj')
    parser.add_argument('--filepattern', default='.obj', help='filepattern')
    args = parser.parse_args()
    convert(args.src, args.dst, args.filepattern)
