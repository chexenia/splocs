import os
import numpy as np
import h5py
import pymesh
import argparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from util import sort_nicely, veclen, filter_reindex, find_rbm_procrustes, transform

def enumerateFiles(path, templates=[]):
    """ Recursively enumerates files in the folder of specific template. Returns a list of file paths
        contaied in those folders
    """
    files = []
    tags = []
    for d in os.listdir(path):
        #full = os.path.normpath(path + os.path.sep + d)
        full = os.path.join(path, d)
        if os.path.isdir(full):
            files.extend(enumerateFiles(full, templates))
        else:
            for t in templates:
                if t in d:
                    files.append(full)
                    break
    return files

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

def save_animation(dstpath, verts, tris, W = None, verts_mean=None, scale=None):
    """save data to hdf5 file"""
    with h5py.File(dstpath, 'w') as f:
        f.create_dataset('verts', data=verts, compression='gzip')
        f['tris'] = tris
        if W is not None:
            f['weights'] = W
        if verts_mean is not None:
            f.attrs['mean'] = verts_mean
        if scale is not None:
            f.attrs['scale'] = scale

def load_animation(srcpath):
    with h5py.File(srcpath, 'r') as f:
        verts = f['verts'].value.astype(np.float)
        tris = f['tris'].value
        weights = None
        mean = None
        scale = None
        if 'weights' in f:
            weights = f['weights'].value
        if 'mean' in f.attrs:
            mean = f.attrs['mean']
        if 'scale' in f.attrs:
            scale = f.attrs['scale']

        print(verts.shape, tris.shape, weights.shape if weights is not None else 'None', mean, scale)
        return verts, tris, weights, mean, scale

def convert_sequence_to_hdf5(src, filepattern, hdf_output_file, normalize):
    verts_all = []
    tris = None
    files = enumerateFiles(src, [filepattern])
    sort_nicely(files)
    for f in files:
        print("loading", f)
        mesh = pymesh.load_mesh(f)
        verts, new_tris = mesh.vertices, mesh.faces
        if tris is not None and new_tris.shape != tris.shape and new_tris != tris:
            raise ValueError("inconsistent topology between meshes of different frames")
        tris = new_tris
        verts_all.append(verts)

    verts_all = np.array(verts_all, np.float32)
    verts_all, tris, verts_mean, verts_scale = preprocess_mesh_animation(verts_all, tris, normalize=normalize)

    save_animation(hdf_output_file, verts_all, tris, None, verts_mean, verts_scale)

    print("saved as %s" % hdf_output_file)

def preprocess_mesh_animation(verts, tris, correct=False, normalize=False):
    """ 
    Preprocess the mesh animation:
        - @correct: removes zero-area triangles and keep only the biggest connected component in the mesh
        - @normalize: normalize animation into -0.5 ... 0.5 cube
    """
    print("vertices:", verts.shape, "triangles:", tris.shape)
    assert(verts.ndim == 3)
    assert(tris.ndim == 2)
    if correct:
        # check for zero-area triangles and filter
        e1 = verts[0, tris[:,1]] - verts[0, tris[:,0]]
        e2 = verts[0, tris[:,2]] - verts[0, tris[:,0]]
        n = np.cross(e1, e2)
        tris = tris[veclen(n) > 1.e-8]
        # remove unconnected vertices
        ij = np.r_[np.c_[tris[:,0], tris[:,1]], 
                np.c_[tris[:,0], tris[:,2]], 
                np.c_[tris[:,1], tris[:,2]]]
        G = csr_matrix((np.ones(len(ij)), ij.T), shape=(verts.shape[1], verts.shape[1]))
        n_components, labels = connected_components(G, directed=False)
        if n_components > 1:
            size_components = np.bincount(labels)
            if len(size_components) > 1:
                print("[warning] found %d connected components in the mesh, keeping only the biggest one", n_components, "component sizes: ", size_components)
            keep_vert = labels == size_components.argmax()
        else:
            keep_vert = np.ones(verts.shape[1], np.bool)
        verts = verts[:, keep_vert, :]
        tris = filter_reindex(keep_vert, tris[keep_vert[tris].all(axis=1)])
    verts_scale = 1.0
    verts_mean = np.zeros(3)
    if normalize:
        # normalize triangles to -0.5...0.5 cube
        verts_mean = verts.mean(axis=0).mean(axis=0)
        verts -= verts_mean
        verts_scale = np.abs(verts.ptp(axis=1)).max()
        verts /= verts_scale
    print("after preprocessing: vertices: ", verts.shape, "triangles: ", verts.shape, "vmean:",verts_mean, "vscale:", verts_scale)
    return verts, tris, verts_mean, verts_scale

def load_splocs(component_hdf5_file):
    with h5py.File(component_hdf5_file, 'r') as f:
        tris = f['tris'].value
        Xmean = f['default'].value
        indices = f['indices'].value
        vmean = f.attrs['vmean']
        vscale = f.attrs['vscale']
        names = sorted(list(set(f.keys()) - set(['tris', 'default', 'indices', 'vmean', 'vscale'])))
        components = np.array([
            f[name].value - Xmean 
            for name in names])
    
    print('splocs loaded:', Xmean.shape, tris.shape, components.shape, names, vmean, vscale)

    return Xmean, tris, components, names, indices, vmean, vscale

def save_splocs(output_sploc_file, Xmean, tris, C, vmean, vscale):
    with h5py.File(output_sploc_file, 'w') as f:
        f['default'] = Xmean
        f['tris'] = tris
        f['indices']= []
        f.attrs['vmean'] = vmean
        f.attrs['vscale'] = vscale
        for i, c in enumerate(C):
            f['comp%03d' % i] = c + Xmean  #TODO: is it necessary??
    print('splocs saved:', Xmean.shape, tris.shape, C.shape, vmean, vscale)

def debug_animation(input_animation_file, filepattern, dst):
    verts, tris, _, _, _ = load_animation(input_animation_file)
    print("sample shape", verts.shape, tris.shape)  
    if not os.path.exists(dst):
        os.makedirs(dst)
    for i in range(verts.shape[0]):
        mesh = pymesh.form_mesh(verts[i, :, :], tris)
        spath = os.path.join(dst, 'sample'+str(i)+filepattern)
        print('saving', spath)
        pymesh.save_mesh(spath, mesh)
 
def align(input_hdf5_file, output_hdf5_file):
    verts, tris, _, mean, scale = load_animation(input_hdf5_file)

    v0 = verts[0]
    verts_new = []
    for i, v in enumerate(verts):
        print("frame %d/%d" % (i+1, len(verts)))
        M = find_rbm_procrustes(v, v0)
        verts_new.append(transform(v, M))
    verts = np.array(verts_new, np.float32)

    save_animation(output_hdf5_file, verts, tris, None, mean, scale)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Import a sequence of mesh files and convert to HDF5 file format')
    parser.add_argument('operation', help='convert or debug or align')
    parser.add_argument('src', 
                        help='path to input')
    parser.add_argument('dst', 
                        help='output hdf5 file path for convert or folder to save debug meshes')
    parser.add_argument('--filepattern',  default='.ply', help='filepattern to load/save')
    parser.add_argument('--normalize',  default=False, help='normalize the data when convert')
    args = parser.parse_args()

    if args.operation == 'convert':
        convert_sequence_to_hdf5(args.src, args.filepattern, args.dst, args.normalize)
    elif args.operation == 'debug':
        debug_animation(args.src, args.filepattern, args.dst)
    elif args.operation == 'align':
        align(args.src, args.dst)
    else:
        raise ValueError('unsupported operation', args.operation)


