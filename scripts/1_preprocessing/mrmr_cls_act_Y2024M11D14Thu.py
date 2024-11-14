# create some pandas data
import sys
sys.path.append('../../src')

import torch
import pandas as pd
import numpy as np

from imuposer.config import Config, amass_datasets
from imuposer.smpl.parametricModel import ParametricModel
from imuposer import math

config = Config(project_root_dir="../../")

# edit >>>
ds_name_ls = ['ACCAD']
act_ls = ['General', 'Running', 'Walking', 'MartialArtsExtended', 'MartialArtsKicks', 'MartialArtsPunches', 'MartialArtsStances', 'MartialArtsWalksTurn']
ds_path = '/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/data/processed_imuposer_act/AMASS/ACCAD'
n_imu = 6
# edit <<<
X = [[] for _ in range(n_imu)]
print('\nX: ', X)
y = []

for ds_name in ds_name_ls:
    amass_dir = config.processed_imu_poser_act / "AMASS" # edit
    amass_dir.mkdir(exist_ok=True, parents=True)
    ds_dir = amass_dir / ds_name

    for act_name in act_ls:
        pose = torch.load(ds_dir / f'{act_name}_pose.pt')
        shape = torch.load(ds_dir / f'{act_name}_shape.pt')
        tran = torch.load(ds_dir / f'{act_name}_tran.pt')
        joint = torch.load(ds_dir / f'{act_name}_joint.pt')
        vrot = torch.load(ds_dir / f'{act_name}_vrot.pt')
        vacc = torch.load(ds_dir / f'{act_name}_vacc.pt')
        print('\npose[0].size():', pose[0].size())
        print('\nshape[0].size():', shape[0].size())
        print('\ntran[0].size():', tran[0].size())
        print('\njoint[0].size():', joint[0].size())
        print('\nvrot[0].size():', vrot[0].size())
        print('\nvacc[0].size():', vacc[0].size())
        '''
        pose[0].size(): torch.Size([359, 24, 3])
        shape[0].size(): torch.Size([10])
        tran[0].size(): torch.Size([359, 3])
        joint[0].size(): torch.Size([359, 24, 3])
        vrot[0].size(): torch.Size([359, 6, 3, 3])
        vacc[0].size(): torch.Size([359, 6, 3])
        '''
        # pose = torch.reshape(pose[0], (-1, 24 * 3))
        # tran = torch.reshape(tran[0], (-1, 3))
        # joint = torch.reshape(joint[0], (-1, 24 * 3))
        # vrot = torch.reshape(vrot[0], (-1, 6 * 3 * 3))
        # vacc = torch.reshape(vacc[0], (-1, 6 * 3))
        # print('\npose.size():', pose.size())
        # print('\ntran.size():', tran.size())
        # print('\njoint.size():', joint.size())
        # print('\nvrot.size():', vrot.size())
        # print('\nvacc.size():', vacc.size())
        '''
        pose.size(): torch.Size([359, 72])
        tran.size(): torch.Size([359, 3])
        joint.size(): torch.Size([359, 72])
        vrot.size(): torch.Size([359, 54])
        vacc.size(): torch.Size([359, 18])
        '''
        vrot_imu_ls = []
        vrot = vrot[0]
        for i in range(vrot.size()[0]):
            act_i = act_ls.index(act_name)
            print('\nact_i: ', act_i)
            y_ = [act_i] * 9 # edit
            print('\ny_: ', y_)
            y.extend(y_)
            
            vrot_ = vrot[0]
            print('\nvrot_.size(): ', vrot_.size())

            for j in range(6):
                vrot_imu = vrot_[j]
                print('\nvrot_imu.size(): ', vrot_imu.size())
                vrot_imu = vrot_imu.flatten()
                print('\nvrot_imu.size(): ', vrot_imu.size())
                # vrot_imu.size():  torch.Size([9])

                X[j].extend(vrot_imu)
                
X_arr = np.rot90(np.array(X))
y_arr = np.array(y)
print('\nnp.shape(X_arr): ', np.shape(X_arr))
print('\nnp.shape(y_arr): ', np.shape(y_arr))
'''
np.shape(X_arr):  (39060, 6)
np.shape(y_arr):  (39060,)
'''
X = pd.DataFrame(X_arr)
y = pd.Series(y_arr)


# X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
# X = pd.DataFrame(X)
# y = pd.Series(y)

# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=6)
print('\nX: ', X)
print('\ny: ', y)
print('\nselected_features: ', selected_features)
'''
Length: 39060, dtype: int64
selected_features:  [0, 4, 1, 2, 3, 5]
'''