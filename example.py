
import sys
sys.path.append("/home/yhu459/envs/amptorch_fork/amptorch/")
import numpy as np
from amptorch.gaussian import SNN_Gaussian
from amptorch.data_preprocess import AtomsDataset, factorize_data, collate_amp, TestDataset
from ase.calculators.emt import EMT
from ase.io import read
from ase import Atoms

from subsample import subsample_traj

# get training data
label = 'test'

distances = np.linspace(2, 5, 100)
images = []
for l in distances:
    image = Atoms(
        "CuCO",
        [
            (-l * np.sin(0.65), l * np.cos(0.65), 0),
            (0, 0, 0),
            (l * np.sin(0.65), l * np.cos(0.65), 0),
        ],
    )
    image.set_cell([10, 10, 10])
    image.wrap(pbc=True)
    image.set_calculator(EMT())
    images.append(image)

# define symmetry functions to be used
Gs = {}
Gs["G2_etas"] = np.logspace(np.log10(0.05), np.log10(5.0), num=4)
Gs["G2_rs_s"] = [0] * 4
Gs["G4_etas"] = [0.005]
Gs["G4_zetas"] = [1.0]
Gs["G4_gammas"] = [+1.0, -1]
Gs["cutoff"] = 6.5

images_keep, idx_keep = subsample_traj(images, Gs)

print(len(idx_keep))

# can used `images_keep` for further analysis
