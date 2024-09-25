import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
from matplotlib import pyplot as plt
import shap
from feature_engineering import get_data  

path = 'tracking_output/data_47091baa.csv'
outpath = 'shap_averaged_GBC.csv'
data_from_csv = pd.read_csv(path)
X, y, indexed = get_data(data_from_csv)

verbose = True
grid_iterations = 2
cv_iterations = 2
sgkf = StratifiedGroupKFold(n_splits=4, shuffle=True)
groups = indexed['file']
idx_file_particle = indexed[['file', 'particle']]
param_grid = {'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1, 10],
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]}


def sh_plot(shap_values, feature_values, feature_names):
    shap.summary_plot(shap_values,
                      feature_values,
                      feature_names=feature_names,
                      sort=False,
                      color_bar=False,
                      plot_size=(10, 10),
                      )


''' GBC grid search '''

pivots = []
baselines = []
for i in range(grid_iterations):
    gbc = GradientBoostingClassifier(n_estimators=1000)
    grid_search = GridSearchCV(gbc, param_grid, cv=sgkf, refit=False)

    grid_search.fit(X, y, groups=groups)

    pivots.append(pd.pivot_table(pd.DataFrame(grid_search.cv_results_),
                                 values='mean_test_score',
                                 index='param_learning_rate',
                                 columns='param_max_depth'))

    baseline = DecisionTreeClassifier(max_depth=1)
    baselines.append(np.mean(cross_val_score(baseline, X, y, cv=sgkf, groups=groups)))

mean_accuracy = pd.concat(pivots).mean(level=0)
sns.heatmap(mean_accuracy, annot=True)
plt.show()
plt.close()

print('Baseline performance: ' + str(np.mean(baselines)))

best_max_depth = mean_accuracy.max().idxmax()
best_learning_rate = mean_accuracy.max(axis=1).idxmax()


''' GBC SHAP explanation '''

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

        gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=best_learning_rate, max_depth=best_max_depth)
        gbc.fit(X_train, y_train)

        explainer = shap.TreeExplainer(gbc)
        shap_values = explainer.shap_values(X_test)

        shap_splits.append(shap_values)
        X_test_splits.append(X_test)
        idx_splits.append(idx_file_particle.loc[test_idxs])

        if verbose:
            plt.title('Single split')
            sh_plot(shap_values, X_test, X.columns)

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

for col in shaps_and_features.columns:
    shap_averaged[col] = shap_repeats[[x for x in shap_repeats.columns if col[1:] == x[1:]]].mean(axis=1)

shap_averaged.to_csv(outpath, index=False)
plt.title('Aggregated from ' + str(cv_iterations) + ' CV repeats')
sh_plot(shap_averaged.iloc[:, 20:].to_numpy(), shap_averaged.iloc[:, :20].to_numpy(), X.columns)
