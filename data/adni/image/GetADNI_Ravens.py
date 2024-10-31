# how to run in CUBIC
# module load python/anaconda/3
# source activate BrainPar
# python 2-Her_Atlas.py 10

import numpy as np
import nibabel as nib
import nilearn
from nilearn import plotting
import scipy
import skimage
import scipy.io
import os
from datetime import datetime
import sys

atlas = nib.load("/gpfs/fs001/cbica/home/user/231110-GIANT/Data/Atlas/BLSA_SPGR+MPRAGE_averagetemplate_muse_seg_DS222.nii.gz")
atlas = atlas.get_fdata()
GW_atlas = atlas>100
G_atlas = atlas==150
W_atlas = atlas==250

tar_path = "/gpfs/fs001/cbica/projects/ADNI/Pipelines/ADNI_3.5D_2020/Protocols/RAVENS/"
sub_names = [ item for item in os.listdir(tar_path) if os.path.isdir(os.path.join(tar_path, item)) ]
G_matrix = np.zeros((len(sub_names), np.sum(G_atlas)))
W_matrix = np.zeros((len(sub_names), np.sum(W_atlas)))
for k in range(len(sub_names)):
    if (k%100==0) | (k==1) | (k==len(sub_names)):
        print("subj " + str(k))
    try:
        v=nib.load("/gpfs/fs001/cbica/projects/ADNI/Pipelines/ADNI_3.5D_2020/Protocols/RAVENS/" +sub_names[k]+ "/" +sub_names[k]+ "_T1_LPS_N4_brain_muse-ss_Mean_fastbc_muse_seg_dramms-0.3_RAVENS_150_s2_DS.nii.gz")
        v = v.get_fdata()
        G_matrix[k,:] = v[G_atlas]
        v=nib.load("/gpfs/fs001/cbica/projects/ADNI/Pipelines/ADNI_3.5D_2020/Protocols/RAVENS/" +sub_names[k]+ "/" +sub_names[k]+ "_T1_LPS_N4_brain_muse-ss_Mean_fastbc_muse_seg_dramms-0.3_RAVENS_250_s2_DS.nii.gz")
        v = v.get_fdata()
        W_matrix[k,:] = v[W_atlas]
    except:
        print("ERROR: subj " + str(k))
        pass
np.save("/gpfs/fs001/cbica/home/user/231110-GIANT/231205-Evaluation/1-out/ADNI_G.npy",G_matrix)
np.save("/gpfs/fs001/cbica/home/user/231110-GIANT/231205-Evaluation/1-out/ADNI_W.npy",W_matrix)
with open("/gpfs/fs001/cbica/home/user/231110-GIANT/231205-Evaluation/1-out/ADNI_subj.txt", 'w') as f:
    for item in sub_names:
        f.write("%s\n" % item)