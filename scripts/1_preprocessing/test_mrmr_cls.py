# create some pandas data
import pandas as pd
from sklearn.datasets import make_classification

ds_path = '/media/brcao/eData4TB1/Repos/IMUPoser_bryanbocao/IMUPoser/data/processed_imuposer_act/AMASS/ACCAD'

X, y = make_classification(n_samples = 1000, n_features = 50, n_informative = 10, n_redundant = 40)
X = pd.DataFrame(X)
y = pd.Series(y)

# select top 10 features using mRMR
from mrmr import mrmr_classif
selected_features = mrmr_classif(X=X, y=y, K=10)
print('\nX: ', X)
print('\ny: ', y)
print('\nselected_features: ', selected_features)