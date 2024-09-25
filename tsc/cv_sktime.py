import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import StratifiedGroupKFold

import warnings
warnings.simplefilter(action='ignore', category=sklearn.exceptions.UndefinedMetricWarning)

from etl_tsc import load_data, get_X_dfX_y_groups, fsets
from normalization import get_standard_scaling, apply_standard_scaling

#logger = None

scoring=['accuracy', 'precision','recall','f1']


def format_scores_df(scores):
    scores = scores.drop(columns=['fit_time','score_time'])
    scores.columns = scores.columns.str.replace('test_','')
    #print(scores.mean())
    #print(scores.std())
    return scores


def cv_single_with_norm(classifier, X, y, groups, cv):
    columns = ['accuracy','precision','recall','f1']
    scores = pd.DataFrame(columns=columns)
    for train_index,val_index in cv.split(X,y,groups):
        # scale training data to mean,std 0,1
        mean,std = get_standard_scaling(X[train_index])
        logger.debug("mean:")
        logger.debug(mean)
        logger.debug("std:")
        logger.debug(std)
        X_train_scaled = apply_standard_scaling(X[train_index],mean,std)
        
        classifier.fit(X_train_scaled, y[train_index])

        # scale validation data to mean,std 0,1
        X_val_scaled = apply_standard_scaling(X[val_index],mean,std)
        pred = classifier.predict(X_val_scaled)

        truth = y[val_index]
        #print('truth')
        #print(truth)
        #print('pred')
        #print(pred)

        # get fold accuracy and append
        fold_acc = accuracy_score(truth, pred)
        fold_prc = precision_score(truth, pred)
        fold_rec = recall_score(truth, pred)
        fold_f1 = f1_score(truth, pred)
        scores.loc[len(scores)] = [fold_acc,fold_prc,fold_rec,fold_f1]

    return scores
    
    
def cv_sgkf(classifier, X, y, groups, repeats=10):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True)
    #print(classifier)
    #print(cv)

    scores_all = []
    for i in range(repeats):
        scores = cv_single_with_norm(classifier, X, y, groups, cv)
        #scores = cross_validate(classifier, X, y, cv=cv, scoring=scoring, groups=groups)
        #scores_all.append(pd.DataFrame.from_dict(scores))
        scores_all.append(scores)
    
    scores = pd.concat(scores_all)
    #scores = format_scores_df(scores)
    scores['cv'] = str(cv)
    
    return scores


# In[ ]:


from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

def run_kneighbors(data, n_neighbors=5, repeats=20):
    logger.info("run_kneighbors")
    classifier = KNeighborsTimeSeriesClassifier(n_neighbors=n_neighbors)

    scores_all = []
    for fset in fsets.keys():
        logger.debug("run_kneighbors: fset: " + fset)
        #X, dfX, y, groups, debug_polar, debug_data = prepare_Xy(data, fset)
        X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

        logger.debug("run_kneighbors: shape X: " + str(X.shape))
        logger.debug("run_kneighbors: shape y: " + str(y.shape))
        logger.debug("run_kneighbors: shape groups: " + str(groups.shape))
        
        scores = cv_sgkf(classifier, X, y, groups, repeats=repeats)
        scores['classifier'] = 'KNeighborsTSC'
        scores['fset'] = fset
        scores_all.append(scores)

    scores = pd.concat(scores_all)
    print("run_kneighbors")
    print(scores.groupby(['cv','fset']).mean())
    print(scores.groupby(['cv','fset']).std())
    print(scores.shape)

    return scores



# In[ ]:


from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sktime.transformations.panel.rocket import Rocket

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run_rocket(data, repeats=20):
    logger.info("run_rocket")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    #classifier = RidgeClassifier()
    #classifier.fit(X_transform, y)


    scores_all = []
    for fset in fsets.keys():
        logger.debug("run_rocket: fset: " + fset)
        #X, dfX, y, groups, debugm, debugn = prepare_Xy(data, fset)
        X, dfX, y, groups, debugm, debugn = get_X_dfX_y_groups(data, fset)

        logger.debug("run_rocket: shape dfX: " + str(dfX.shape))
        logger.debug("run_rocket: shape y: " + str(y.shape))
        logger.debug("run_rocket: shape groups: " + str(groups.shape))
        rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
        rocket.fit(dfX)
        X_transform = rocket.transform(dfX)
        logger.debug("run_rocket: shape X_transform: " + str(X_transform.shape))
        X_np = X_transform.to_numpy()
        print("X_transform")
        print(X_transform)
        logger.debug("run_rocket: shape X_np: " + str(X_np.shape))
        print("X_np")
        print(X_np)

        scores = cv_sgkf(classifier, X_np, y, groups, repeats=repeats)
        scores['classifier'] = 'Rocket'
        scores['fset'] = fset
        scores_all.append(scores)

    scores = pd.concat(scores_all)
    
    print("run_rocket")
    print(scores.groupby(['cv','fset']).mean())
    print(scores.groupby(['cv','fset']).std())
    print(scores.shape)

    return scores



# 
# 
# 1.   Stratified split (add function, with loop)
# 2.   Infinity loop to check epoch effect for inceptiontime
# 3.   SHAP for NN, TSC
# 4.   plot acc (mean, std)
# 
# 

# In[ ]:



import click
import pandas as pd
from pathlib import Path

from utility import parse_config, set_logger

@click.command()
#@click.argument("config_file", type=str, default="config.yml")
@click.option("--paths", type=str, default="paths.yml")
@click.option("--config", type=str, default="config.yml")
@click.option("--job_name", type=str, default="cv_sktime")
@click.option("--job_id", type=str)
def cv_sktime(paths, config, job_name, job_id):
    paths = parse_config(paths)
    config = parse_config(config)

    log_dir = paths["log"]["tsc"]
    
    # configure logger
    global logger
    logger = set_logger(log_dir + "/" + job_name + ".log")

    # Load config from config file
    logger.info(f"Load config from {config}")


    # read the data 
    data_dir = paths["data"]["dir"]
    raw_data_file = paths["data"]["raw_data_file"]

    data = load_data(Path(data_dir) / raw_data_file)
    logger.info('Loaded data shape: ' + str(data.shape))


    scores_all = []
    
    # run KNeighbors
    n_neighbors = config["kneighbors"]["n_neighbors"]
    repeats = config["kneighbors"]["repeats"]
    scores = run_kneighbors(data, n_neighbors, repeats)
    scores_all.append(scores)

    # run Rocket
    repeats = config["rocket"]["repeats"]
    scores = run_rocket(data, repeats)
    scores_all.append(scores)


    # store output
    output_dir = paths["output"]["cv"]
    output_csv = job_name + "-" + job_id + ".csv"
    output_csv = Path(output_dir) / output_csv
    
    scores = pd.concat(scores_all)
    scores.to_csv(output_csv, index=False)
    logger.info("Wrote scores to " + str(output_csv))
    
    
if __name__ == "__main__":
    cv_sktime()

    
