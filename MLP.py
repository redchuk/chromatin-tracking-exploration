import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from feature_engineering import get_data  

import tensorflow as tf
# https://stackoverflow.com/questions/66814523/shap-deepexplainer-with-tensorflow-2-4-error
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras import layers
from tensorflow.keras import models

path = 'tracking_output/data_47091baa.csv'  
outpath = 'shap_averaged_MLP.csv'
data_from_csv = pd.read_csv(path)
frame_counts = data_from_csv.set_index(['file', 'particle']).index.value_counts()
less_30frames = frame_counts[frame_counts < 30]
data = data_from_csv.set_index(['file', 'particle']).drop(less_30frames.index)

X, y, indexed = get_data(data.reset_index())

verbose = True
cv_iterations = 2
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
groups = indexed['file']
idx_file_particle = indexed[['file', 'particle']]


def sh_plot(shap_values, feature_values, feature_names):
    shap.summary_plot(shap_values,
                      feature_values,
                      feature_names=feature_names,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      )


training_iteration = 0
validation_profiles = pd.DataFrame()
shap_repeats = pd.DataFrame()
shap_averaged = pd.DataFrame()

for i in range(cv_iterations):
    shap_splits = []
    X_test_splits = []
    idx_splits = []

    for train_idxs, test_idxs in sgkf.split(X, y, groups):
        X_train = X.loc[train_idxs]
        X_test = X.loc[test_idxs]
        y_train = y.loc[train_idxs]
        y_test = y.loc[test_idxs]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_norm = scaler.transform(X_train)
        X_test_norm = scaler.transform(X_test)

        input_layer = layers.Input(shape=(20,))
        layer1 = layers.Dense(256, activation='relu')(input_layer)
        layer2 = layers.Dense(256, activation='relu')(layer1)
        layer3 = layers.Dense(256, activation='relu')(layer2)
        layer4 = layers.Dense(256, activation='relu')(layer3)
        layer5 = layers.Dense(256, activation='relu')(layer4)
        layer6 = layers.Dense(256, activation='relu')(layer5)
        layer7 = layers.Dense(256, activation='relu')(layer6)
        output_layer = layers.Dense(1, activation="sigmoid")(layer7)
        model = models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(X_train_norm,
                            y_train,
                            epochs=200,
                            validation_data=(X_test_norm, y_test),
                            verbose=0,
                            )

        validation_profiles['val_acc' + str(training_iteration)] = history.history['val_acc']
        training_iteration += 1

        explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), X_train_norm)
        shap_values = explainer.shap_values(X_test_norm)

        shap_splits.append(shap_values[0])
        X_test_splits.append(X_test)
        idx_splits.append(idx_file_particle.loc[test_idxs])

        if verbose:
            plt.title('Single split')
            sh_plot(shap_values[0], X_test, X.columns)

    all_X_test_splits = pd.concat(X_test_splits)
    all_shap_splits = np.concatenate(shap_splits)
    all_shap_splits_df = pd.DataFrame(all_shap_splits, columns=all_X_test_splits.columns).add_prefix('shap_')
    all_idx_splits = pd.concat(idx_splits)

    if verbose:
        plt.title('Aggregated from 4CV splits')
        sh_plot(all_shap_splits, all_X_test_splits, X.columns)

    list_to_concat = [all_X_test_splits.reset_index(),
                      all_shap_splits_df.reset_index(),
                      all_idx_splits.reset_index()]

    shaps_and_features = pd.concat(list_to_concat, axis=1) \
        .drop('index', axis=1).set_index(['file', 'particle']).add_prefix(str(i) + 'r_')

    shap_repeats = shap_repeats.join(shaps_and_features) if not shap_repeats.empty else shaps_and_features

plt.plot(validation_profiles.mean(axis=1))
plt.show()
plt.close()

for col in shaps_and_features.columns:
        shap_averaged[col] = shap_repeats[[x for x in shap_repeats.columns if col[1:] == x[1:]]].mean(axis=1)

shap_averaged.to_csv(outpath, index=False)
plt.title('Aggregated from ' + str(cv_iterations) + ' CV repeats')
sh_plot(shap_averaged.iloc[:, 20:].to_numpy(), shap_averaged.iloc[:, :20].to_numpy(), X.columns)
