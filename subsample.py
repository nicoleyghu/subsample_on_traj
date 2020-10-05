from amptorch.gaussian import SNN_Gaussian
from amptorch.data_preprocess import AtomsDataset
import numpy as np
from customizedNNSubsampling import subsampling

def subsample_traj(traj, Gs, cutoff_sig=0.05, rate=0.3, method='pykdtree'):
    """
    This function calculates the fingerprints by calling AMPTorch's AtomsDataset object, and used a subsampling algorithm based on k-nearest neighbor. 

    Parameters
    --------------
    traj: list of ase trajectory objects. 
    Gs: a dictionary of the hyperparameters to generate Gaussian symmetry functions
    cutoff_sig: Float. cutoff significance. the cutoff distance equals to the Euclidean norm of the standard deviations in all dimensions of the data points. 
    rate [0.3]: Float. possibility of deletion. 
    method ["pykdtree"]: String. which backend nearest neighbour model to use. possible choices: ["pykdtree", "nmslib", "sklearn", "scipy", "annoy", "flann"]

    Return
    --------------
    images_keep : the result list of subsampled images in the provided ase trajectory
    idx_keep : the list of indices in the provided trajectory that get selected by subsampling algorithm
    """
    training_data = AtomsDataset(traj, SNN_Gaussian, Gs, forcetraining=False,
        label='test', cores=1, delta_data=None)
    fps = training_data.fingerprint_dataset
    # reorganize fps into n_central_atoms_each_frame * k_fingerprints)
    fps_reorganized = []
    index_list = []
    # iterate over frames
    for i, tmp in enumerate(fps):
        for _, fp in tmp:
                fps_reorganized.append(fp)
                index_list.append(i)
    fps_array = np.asarray(fps_reorganized)
    print(fps_array.shape)
    
    # call subsampling algorithm to obtain indexes to keep
    _, idx_keep = subsampling(fps_array, image_index=index_list, \
                                cutoff_sig=cutoff_sig, \
                                rate=rate, method=method)
    del _
    # remove duplicates
    idx_keep = list(set(idx_keep))
    # obtain images to keep
    images_keep = [traj[_] for _ in idx_keep]
    return images_keep, idx_keep

