from pathlib import Path
import numpy as np
import shap
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
# https://stackoverflow.com/questions/66814523/shap-deepexplainer-with-tensorflow-2-4-error
#from tensorflow.compat.v1.keras.backend import get_session
#tf.compat.v1.disable_v2_behavior() 

from utility import parse_config
from etl_tsc import load_data, get_X_dfX_y_groups, fsets

import logging
logger = logging.getLogger(__name__)


'''
This method was copied and modified from 'prepare_data' in InceptionTime/main.py:
https://github.com/hajaalin/InceptionTime/blob/f3fd6c5e9298ec9ca5d0fc594bb07dd1decc3718/main.py#L15
'''
def prepare_data_for_inception(X,y):
    # use all data, use cross-validation
    x_train = X.copy()
    y_train = y.copy()

    nb_classes = len(np.unique(np.concatenate((y_train,), axis=0)))

    # make the min to zero of labels
    #y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, nb_classes, y_true_train, enc


from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_score


def standard_scale_x_by_series(_X):
    scaled = np.ndarray(_X.shape)

    # loop over samples
    for i in range(_X.shape[0]):
        # loop over features
        for j in range(_X.shape[1]):
            # calculate mean and std for the time series
            u = np.mean(_X[i][j])
            s = np.std(_X[i][j])
            # scale the time series
            scaled[i][j] = (_X[i][j] - u) / s
            #print(np.mean(scaled[i][j]))
            #print(np.std(scaled[i][j]))

    return scaled


def get_standard_scaling(X):
    logger.debug("get_standard_scaling")
    logger.debug(X.shape)
    # X dimensions:
    # i: samples, ~200 series
    # j: features, ~3-10 features
    # k: the time series, ~30 steps
    mean = np.mean(X,axis=(0,2))
    std = np.std(X, axis=(0,2))

    logger.debug(mean.shape)
    logger.debug(std.shape)
    
    return mean,std

def apply_standard_scaling(X,mean,std):
    logger.debug("apply_standard_scaling")
    logger.debug(X.shape)
    # X dimensions:
    # i: samples, ~200 series
    # j: features, ~3-10 features
    # k: the time series, ~30 steps

    # transpose X so that features are on the trailing index
    Xt = np.transpose(X,(0,2,1))
    logger.debug(Xt.shape)

    # transpose mean and std so that features are on the trailing index
    mean = np.transpose(mean)
    std = np.transpose(std)
    logger.debug(mean.shape)
    logger.debug(std.shape)
    # trust numpy broadcasting
    Xt = Xt - mean
    Xt = Xt / std

    Xt = np.transpose(Xt,(0,2,1))
    logger.debug(Xt.shape)

    return Xt


## Look for a high-performing model, they explain why the average is so high.
## But what is it about that split that makes it so successful?
TARGET_ACCURACY = 0.98

'''
Single cross-validation run
'''
def inceptiontime_cv(cv, X_inc, y_inc, y_true, groups, output_it, \
                     kernel_size, epochs=250, feature_names=None, nb_classes=2, return_model_eval=False, save_shap_values=False):
    output_directory = output_it
    input_shape = X_inc.shape[1:]
    verbose = False

    from classifiers import inception
    classifier_keras = None
    classifier_keras = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, \
                                                      kernel_size=kernel_size, nb_epochs=epochs, \
                                                      verbose=verbose)
    def create_model():
        #print(classifier_keras.model)
        #classifier_keras.model.summary()
        return classifier_keras.model

    batch_size = int(min(X_inc.shape[0] / 10, 16))
    columns = ['accuracy','precision','recall','f1']
    scores = pd.DataFrame(columns=columns)

    shap_deep_list = []
    shap_grad_list = []
    X_test_list = []
    accuracy_list = []
    train_index_list = []
    
    # One-hot encoding is a problem for StratifiedGroupKFold,
    # split using y_true
    for train_index,val_index in cv.split(X_inc,y_true,groups):
        #print('cv loop')
        #print(train_index.shape)
        #print(X_inc[train_index].shape)
        input_shape = X_inc[train_index].shape[1:]
        #print(input_shape)

        #print(train_index)
        #print(val_index)
        #continue
        #print(y_true[train_index])
        #print(y_inc[train_index])
        #break

        # scale training data to mean,std 0,1
        #X_train_scaled = standard_scale_x_by_series(X_inc[train_index])
        mean,std = get_standard_scaling(X_inc[train_index])
        logger.debug("mean:")
        logger.debug(mean)
        logger.debug("std:")
        logger.debug(std)
        X_train_scaled = apply_standard_scaling(X_inc[train_index],mean,std)
        #print(X_train_scaled.shape)
        #logger.debug("scaled2")
        #logger.debug(X_train_scaled)
       
        classifier = KerasClassifier(model=create_model(), \
                                     epochs=epochs, \
                                     batch_size=batch_size, \
                                     verbose=verbose)
        classifier.fit(X_train_scaled, y_inc[train_index])

        # scale validation data to mean,std 0,1
        #X_val_scaled = standard_scale_x_by_series(X_inc[val_index])
        X_val_scaled = apply_standard_scaling(X_inc[val_index],mean,std)
        pred = classifier.predict(X_val_scaled)

        truth = y_true[val_index]

        #print('truth')
        #print(truth)
        #print('pred')
        #print(pred)

        # prediction is onehot-encoded, reverse it
        pred = pred.argmax(1)
        #print(pred)

        # get fold accuracy and append
        fold_acc = accuracy_score(truth, pred)
        fold_prc = precision_score(truth, pred)
        fold_rec = recall_score(truth, pred)
        fold_f1 = f1_score(truth, pred)
        scores.loc[len(scores)] = [fold_acc,fold_prc,fold_rec,fold_f1]

        model_eval = None
        # if accuracy is close enough to target,
        # save model and SHAP values
        print(fold_acc)
        if return_model_eval and abs(TARGET_ACCURACY - fold_acc) < 0.02:
            print("sufficiently accurate model found")
            (shap_deep, shap_grad) = get_shap_values(classifier.model_, \
                                                     X_train_scaled, \
                                                     X_val_scaled)
    
            model_eval = (classifier.model_, feature_names, shap_deep, shap_grad, X_val_scaled, pred, truth)
            return pd.DataFrame(), model_eval, shap_lists

        accuracy_list.append(fold_acc)
        train_index_list.append(train_index)
        if save_shap_values:
            (shap_deep, shap_grad) = get_shap_values(classifier.model_, \
                                                     X_train_scaled, \
                                                     X_val_scaled)
            shap_deep_list.append(shap_deep[0])
            shap_grad_list.append(shap_grad[0])
            X_test_list.append(X_val_scaled)

               
    scores['classifier'] = 'InceptionTime'
    scores['kernel_size'] = kernel_size
    scores['epochs'] = epochs

    print("len(shap_deep_list):%d" % len(shap_deep_list))
    print("len(shap_grad_list):%d" % len(shap_grad_list))
    
    shap_lists = (shap_deep_list, shap_grad_list, X_test_list, accuracy_list, train_index_list)

    return scores, model_eval, shap_lists


def get_shap_values(model_, X_train, X_test):
    # SHAP
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
    explainer = shap.DeepExplainer((model_.layers[0].input, \
                                    model_.layers[-1].output), \
                                   X_train)
    shap_values_deep = explainer.shap_values(X_test)

    explainer = shap.GradientExplainer(model_, X_train)
    #explainer.expected_value = explainer.expected_value[0]
    shap_values_grad = explainer.shap_values(X_test)

    return (shap_values_deep, shap_values_grad)
   

'''
Repeat cross-validation
'''
def inceptiontime_cv_repeat(data, output_it, fset, kernel_size=20, epochs=250, repeats=10,job_id='', return_model_eval=False, save_shap_values=False):
    logger.info(fset)
    X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

    # fset includes also class and file, get feature names for SHAP
    feature_names = dfX.columns
    print(feature_names)

    # prepare_data_for inception returns all, no split to train and test sets
    X_inc, y_inc, nb_classes, y_true, enc = prepare_data_for_inception(X,y)

    logger.debug("X_inc: " + str(X_inc.shape))
    logger.debug("y_inc: " + str(y_inc.shape))

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)

    scores_all = []
    shap_lists_all = []
    for i in range(repeats):
        print('repeat: %d/%d' % (i+1, repeats))
        logger.debug('repeat: %d/%d' % (i+1, repeats))
        cv_output = inceptiontime_cv(cv, X_inc, y_inc, y_true, \
                                     groups, output_it, \
                                     kernel_size=kernel_size, epochs=epochs, \
                                     feature_names=feature_names, \
                                     nb_classes=nb_classes, \
                                     return_model_eval=return_model_eval, \
                                     save_shap_values=save_shap_values)
        scores, model_eval, shap_lists = cv_output
        shap_lists_all.append(shap_lists)

        scores['repeat'] = i+1
        scores_all.append(scores)

        #print(return_model_eval)
        #print(model_eval)
        if return_model_eval and (model_eval != None):
            return model_eval
        
    scores = pd.concat(scores_all)

    scores['cv'] = str(cv)
    scores['fset'] = fset
    scores['kernel_size'] = kernel_size
    scores['epochs'] = epochs
    scores['job_id'] = job_id

    logger.info(f"inceptiontime_cv_repeat: %s feature_set:%s, kernel_size:%d, epochs:%d, accuracy:%f0.00" % (str(cv), fset, kernel_size, epochs, scores['accuracy'].mean()))

    print("len(shap_lists_all):%d" % len(shap_lists_all))
    return scores, shap_lists_all


import click
from datetime import datetime
import pandas as pd
import sys
import time

@click.command()
@click.option("--inceptiontime_dir", type=str)
@click.option("--paths", type=str, default="paths.yml")
@click.option("--kernel_size", type=int, default=20)
@click.option("--epochs", type=int, default=100)
@click.option("--fset", type=click.Choice(fsets.keys()), default="f_mot_morph")
@click.option("--repeats", type=int, default=20)
@click.option("--save_shap_values", is_flag=True, default=False)
@click.option("--job_name", type=str, default="tsc_it")
@click.option("--job_id", type=str)
@click.option("--now", type=str)
def cv_inceptiontime(inceptiontime_dir, paths, kernel_size, epochs, fset, repeats, save_shap_values, job_name, job_id, now):
    paths = parse_config(paths)

    log_dir = Path(paths["log"]["tsc"]) / job_name / now
    log_dir.mkdir(parents=True, exist_ok=True)

    if not now:
        now = datetime.now().strftime("%Y%m%d%H%M%S")

    # configure logger
    log_file = log_dir / (f"cv_inceptiontime_%s_%s_%s.log" % (job_name, now, job_id))
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_file, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    print("Logging to " + str(log_file))

    # log the version of this code
    logger.info(Path(__file__).absolute())

    # add InceptionTime source to Python path
    if inceptiontime_dir == 'TEST':
        # this is test mode, without cloning code from GitHub
        inceptiontime_dir = paths["src"]["inceptiontime"]
    sys.path.insert(1, inceptiontime_dir)

    # log the InceptionTime version
    logger.info(inceptiontime_dir)

    # output folders
    output_cv = Path(paths["output"]["cv"]) / job_name / now
    output_cv.mkdir(parents=True, exist_ok=True)
    output_it = Path(paths["output"]["it"]) / job_id
    output_it.mkdir(parents=True, exist_ok=True)
    output_it = str(output_it) + "/"
    output_shap = Path(paths["output"]["shap"]) / job_name / now / job_id
    output_shap.mkdir(parents=True, exist_ok=True)

    # read the data 
    data_dir = paths["data"]["dir"]
    raw_data_file = paths["data"]["raw_data_file"]

    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))


    tic = time.perf_counter()
    
    logger.info("single fset")
    scores, shap_lists_all = inceptiontime_cv_repeat(data, output_it, fset, kernel_size=kernel_size, epochs=epochs, repeats=repeats, save_shap_values=save_shap_values, job_id=job_id)
        
    toc = time.perf_counter()
    logger.info(f"Finished processing in {(toc-tic) / 60:0.1f} minutes.")
            
    scores_file = "cv_" + job_name + "_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
    scores_file = output_cv / scores_file
    scores.to_csv(scores_file, index=False)
    logger.info("Wrote scores to " + str(scores_file))


    # save X_test and accuracy lists, even if SHAP values are not calculated
    shap2npy(fset, shap_lists_all, output_shap)


def shap2npy(fset, shap_lists_all, output_shap):
    shap_deep_list = []
    shap_grad_list = []
    X_test_list = []
    accuracy_list = []
    train_index_list = []

    # loop over repetitions
    for shap_lists in shap_lists_all:
        # each repetition produced a tuple of lists
        shapd, shapg, X_test, acc, train = shap_lists
        
        shap_deep_list.extend(shapd)
        shap_grad_list.extend(shapg)
        X_test_list.extend(X_test)
        accuracy_list.extend(acc)
        train_index_list.extend(train)

    np.save(output_shap / 'shap_deep_list.npy', shap_deep_list)
    np.save(output_shap / 'shap_grad_list.npy', shap_grad_list)
    np.save(output_shap / 'X_test_list.npy', X_test_list)
    np.save(output_shap / 'accuracy_list.npy', accuracy_list)
    np.save(output_shap / 'train_index_list.npy', train_index_list)
    np.save(output_shap / 'features.npy', fsets[fset])
        
        

if __name__ == "__main__":
    cv_inceptiontime()
    

