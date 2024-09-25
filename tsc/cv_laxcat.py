import keras
from pathlib import Path
import numpy as np
import sklearn
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score

from cv_inceptiontime import get_standard_scaling, apply_standard_scaling
from etl_tsc import load_data, get_X_dfX_y_groups, fsets
from laxcat import Classifier_LAXCAT
from utility import parse_config

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

    # save original y because later we will use binary
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train,), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, nb_classes, y_true_train, enc




    

'''
Single cross-validation run
'''
def cv_single_run(classifier_name, params, \
                  cv, X_inc, y_inc, y_true, groups, output_it, \
                  epochs=250, nclasses=2):
    output_directory = output_it
    input_shape = X_inc.shape[1:]
    verbose = False

    # this import is done here after __main__ has added src directory to path
    classifier_keras = None

    if classifier_name == 'InceptionTime':
        from classifiers import inception

        kernel_size = params['kernel_size']
        classifier_keras = inception.Classifier_INCEPTION(output_directory, \
                                                          input_shape, \
                                                          nclasses, \
                                                          kernel_size=kernel_size, \
                                                          nb_epochs=epochs, \
                                                          verbose=verbose)
    elif classifier_name == 'LAXCAT':
        classifier_keras = Classifier_LAXCAT(params, input_shape)
        
    def create_model():
        #print(classifier_keras.model)
        #classifier_keras.model.summary()
        return classifier_keras.model

    batch_size = int(min(X_inc.shape[0] / 10, 16))
    columns = ['accuracy','precision','recall','f1']
    scores = pd.DataFrame(columns=columns)

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
        print(X_train_scaled.shape)
        #logger.debug("scaled2")
        #logger.debug(X_train_scaled)
        
        classifier = KerasClassifier(model=create_model, \
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
    

    return scores


'''
Repeat cross-validation
'''
def cv_repeat(classifier_name, params, data, output_it, fset, \
              epochs=250, repeats=10, job_id=''):
    logger.info(fset)
    X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

    # prepare_data_for inception returns all, no split to train and test sets
    X_inc, y_inc, nclasses, y_true, enc = prepare_data_for_inception(X,y)

    logger.debug("X_inc: " + str(X_inc.shape))
    logger.debug("y_inc: " + str(y_inc.shape))

    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)

    scores_all = []
    for i in range(repeats):
        logger.debug('repeat: %d/%d' % (i+1, repeats))
        scores = cv_single_run(classifier_name, params, \
                               cv, X_inc, y_inc, y_true, groups, output_it, \
                               epochs=epochs, nclasses=nclasses)
        scores['repeat'] = i+1
        scores_all.append(scores)
    scores = pd.concat(scores_all)

    scores['classifier'] = classifier_name
    for p in params.keys():
        scores[p] = params[p]
    scores['cv'] = str(cv)
    scores['fset'] = fset
    scores['epochs'] = epochs
    scores['job_id'] = job_id

    logger.info(f"inceptiontime_cv_repeat: %s feature_set:%s, params:%d, epochs:%d, accuracy:%f0.00" % (str(cv), fset, str(params), epochs, scores['accuracy'].mean()))
    
    return scores


import click
from datetime import datetime
import pandas as pd
import sys
import time

@click.command()
@click.option("--paths", type=str, default="paths.yml")
@click.option("--kernel_size", type=int, default=20)
@click.option("--nfilters", type=int, default=32)
@click.option("--epochs", type=int, default=100)
@click.option("--fset", type=click.Choice(["f_mot","f_mot_morph","f_mot_morph_dyn", "f_mot_morph_dyn_2"]), default="f_mot_morph")
@click.option("--loop_fsets", is_flag=True, default=False)
@click.option("--repeats", type=int, default=20)
@click.option("--job_name", type=str, default="cv_laxcat")
@click.option("--job_id", type=str)
@click.option("--now", type=str)
def cv_laxcat(paths, kernel_size, nfilters, epochs, fset, loop_fsets, repeats, job_name, job_id, now):
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


    # add InceptionTime source to Python path
    src_inceptiontime = paths["src"]["inceptiontime"]
    sys.path.insert(1, src_inceptiontime)

    # output folders
    output_cv = Path(paths["output"]["cv"]) / job_name / now
    output_cv.mkdir(parents=True, exist_ok=True)
    output_it = Path(paths["output"]["it"]) / job_id
    output_it.mkdir(parents=True, exist_ok=True)
    output_it = str(output_it) + "/"

    # read the data 
    data_dir = paths["data"]["dir"]
    raw_data_file = paths["data"]["raw_data_file"]
    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))


    # model parameters
    params = {'kernel_size':kernel_size, 'nfilters':nfilters}
    
    tic = time.perf_counter()
    
    scores_all = []

    if loop_fsets:
        logger.info("loop feature sets")

        for f in fsets.keys():
            scores = cv_repeat("LAXCAT", params, \
                               data, output_it, f, epochs=epochs, repeats=repeats, \
                               job_id=job_id)
            scores_all.append(scores)

    else:
        logger.info("single fset")
        scores = cv_repeat("LAXCAT", params, \
                           data, output_it, fset, epochs=epochs, repeats=repeats, \
                           job_id=job_id)
        scores_all.append(scores)
        
    toc = time.perf_counter()
    logger.info(f"Finished processing in {(toc-tic) / 60:0.1f} minutes.")
            
    scores = pd.concat(scores_all)

    scores_file = "cv_" + job_name + "_k" + str(kernel_size) + "_e" + str(epochs) + "_" + now + ".csv"
    scores_file = output_cv / scores_file
    scores.to_csv(scores_file, index=False)
    logger.info("Wrote scores to " + str(scores_file))

if __name__ == "__main__":
    cv_laxcat()
    

