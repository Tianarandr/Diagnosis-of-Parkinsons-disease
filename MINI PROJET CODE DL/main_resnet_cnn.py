from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import MAX_PROTOTYPES_PER_CLASS
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES

from utils.utils import read_all_datasets
from utils.utils import calculate_metrics
from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import plot_pairwise
from utils.utils import transforme_data


from augment import augment_train_set
from resnet import Classifier_RESNET
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
from sklearn import decomposition
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def augment_function(augment_algorithm_name, x_train, y_train, classes, N, limit_N=True):
    if augment_algorithm_name == 'as_dtw_dba_augment':
        return augment_train_set(x_train, y_train, classes, N,limit_N = limit_N,
                                 weights_method_name='as', distance_algorithm='dtw'), 'dtw'



def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)


root_dir = '/Users/tahirintsoa/DATA_PROJET/dba-python/'
root_dir_output = root_dir + 'results/'
root_deep_learning_dir = '/Users/tahirintsoa/DATA_PROJET/dl-tsc/'
root_dir_dataset_archive = '/Users/tahirintsoa/DATA_PROJET/dl-tsc/archives/'

do_data_augmentation = True
do_ensemble = False

if do_ensemble:
    root_dir_output = root_deep_learning_dir + 'results/ensemble/'
else:
    if do_data_augmentation:
        root_dir_output = root_deep_learning_dir + 'results/resnet_augment/'
    else:
        root_dir_output = root_deep_learning_dir + 'results/resnet/'


###################################################################################################################################
# DATA 
###################################################################################################################################
# select dataset GaCo / JuCo / SiCo
# select dataset GaPt / JuPt / SiPt
#transforme_data('Ga')

df_co = pd.read_csv('/Users/tahirintsoa/DATA_PARKINSON/Transformed_Data_ok/Si/Transformed_Co.csv', index_col = 0)
df_pt = pd.read_csv('/Users/tahirintsoa/DATA_PARKINSON/Transformed_Data_ok/Si/Transformed_Pt.csv' , index_col = 0)

print("Shape of DF_co:", df_co.shape)
print("Shape of DF_pt:", df_pt.shape)
print(df_co.shape, df_pt.shape)

df_co_len = df_co.shape[0]
df_pt_len = df_pt.shape[0]

df_co_pca = pd.DataFrame(df_co)
df_pt_pca = pd.DataFrame(df_pt)
y1 = pd.Series([0]*df_co_len)
y2 = pd.Series([1]*df_pt_len, index = range(df_co_len-1,(df_co_len + df_pt_len)-1))

y = pd.concat([y1,y2]) 
y = y.ravel()
X = pd.concat([df_co_pca, df_pt_pca])

print("Shape of X et y : ", X.shape, y.shape)

X_train, X_test, y_train1, y_test1 = train_test_split(X, y, test_size=0.3, shuffle=True)
y_train = pd.DataFrame(y_train1)
y_test = pd.DataFrame(y_test1)

scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

x_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
x_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()
y_train = y_train.ravel()
y_test = y_test.ravel()
# specify the output directory for this experiment
output_dir = '/Users/tahirintsoa/DATA_PARKINSON/Output2/'
print("X_TRAIN :", x_train.shape, y_train.shape)
print("X_TEST :", x_test.shape, y_test.shape)

_, classes_counts = np.unique(y_train, return_counts=True)
nb_prototypes = classes_counts.max()

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

y_train, y_test = transform_labels(y_train, y_test)
classes, classes_counts = np.unique(y_train, return_counts=True)
nb_prototypes = classes_counts.max()

print("Nombre de classe:", classes)
print("Nombre de classe:", nb_classes)
print("max_prototypes:", nb_prototypes)

if do_ensemble==False:
    input_shape = x_train.shape[1:]
    #cnn, mcnn, resnet
    classifier_name = 'mcdcnn'
    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_dir, verbose=True)
    if do_data_augmentation:
        syn_train_set, distance_algorithm = augment_function('as_dtw_dba_augment',
                                                                     x_train, y_train, classes,
                                                                     nb_prototypes,limit_N=False)
        syn_x_train, syn_y_train = syn_train_set
        aug_x_train = np.array(x_train.tolist() + syn_x_train.tolist())
        aug_y_train = np.array(y_train.tolist() + syn_y_train.tolist())

        print(np.unique(y_train,return_counts=True))
        print(np.unique(aug_y_train,return_counts=True))
        print("X_TRAIN Augmenter:", aug_x_train.shape, aug_y_train.shape)
        #y_true = np.argmax(y_test, axis=1)
        y_pred = classifier.fit(aug_x_train, aug_y_train, x_test, y_test, y_test)

        print("Y prediction : ", y_pred)

    else:
        print("NO AUGMENTATION")

        y_pred = classifier.fit(x_train, y_train, x_test,
                                    y_test)

    print('DONE')
    create_directory(output_dir+'DONE')

else:
    from ensemble import Classifier_ENSEMBLE
    classifier_ensemble = Classifier_ENSEMBLE(output_dir, x_train.shape[1:], nb_classes, False)
    classifier_ensemble.fit(x_test, y_test)


