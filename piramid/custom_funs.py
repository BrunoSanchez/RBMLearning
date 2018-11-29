
import numpy as np
import pandas as pd

from astropy.stats import sigma_clipped_stats


from sklearn import preprocessing
from sklearn import decomposition
from sklearn import feature_selection
from sklearn import svm
from sklearn import metrics
from sklearn import neighbors
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold

from rfpimp import *

# We're going to be calculating memory usage a lot,
# so we'll create a function to save us some time!
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


def optimize_df(df):
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')

    #print(mem_usage(df_int))
    #print(mem_usage(converted_int))

    compare_ints = pd.concat([df_int.dtypes,converted_int.dtypes],axis=1)
    compare_ints.columns = ['before','after']
    compare_ints.apply(pd.Series.value_counts)

    df_float = df.select_dtypes(include=['float'])
    converted_float = df_float.apply(pd.to_numeric,downcast='float')

    #print(mem_usage(df_float))
    #print(mem_usage(converted_float))

    compare_floats = pd.concat([df_float.dtypes,converted_float.dtypes],axis=1)
    compare_floats.columns = ['before','after']
    compare_floats.apply(pd.Series.value_counts)

    optimized_df = df.copy()

    optimized_df[converted_int.columns] = converted_int
    optimized_df[converted_float.columns] = converted_float

    mem_df = mem_usage(df)
    mem_op_df = mem_usage(optimized_df)
    print(mem_df)
    print(mem_op_df)
    if mem_df<=mem_op_df:
        print('Memory increased, returning original')
        return df

    return optimized_df


def binning_res(data, bins, return_center_bins=False):
    mean = np.zeros_like(bins[:-1])
    stdv = np.zeros_like(bins[:-1])
    sqrtn= np.zeros_like(mean)
    mean_sim_mag = np.zeros_like(mean)
    for i_bin, low in enumerate(bins[:-1]):
        high = bins[i_bin+1]
        f1data = data[data['mag']<90]
        fdata = f1data[(f1data['sim_mag'] < high) & (f1data['sim_mag'] >= low)]
        fdata_mag = fdata['mag'] - fdata['sim_mag']
        if len(fdata) is 0:
            sqrtn[i_bin] = 0
            mean[i_bin] = np.mean(fdata_mag)
            stdv[i_bin] = np.std(fdata_mag)
            continue
        mean_sim_mag[i_bin] = (high+low)/2.
        sqrtn[i_bin] = np.sqrt(len(fdata_mag))
        mean[i_bin] = np.mean(fdata_mag)
        stdv[i_bin] = np.std(fdata_mag)
    if return_center_bins:
        return bins[:-1]+(high-low)*0.5, mean, stdv, sqrtn
    else:
        return mean, stdv, sqrtn, mean_sim_mag


def binning_robust(data, bins, return_center_bins=False):
    mean = np.zeros_like(bins[:-1])
    stdv = np.zeros_like(mean)
    sqrtn= np.zeros_like(mean)
    mean_sim_mag = np.zeros_like(mean)

    for i_bin, low in enumerate(bins[:-1]):
        high = bins[i_bin+1]
        mean_sim_mag[i_bin] = (high+low)/2.
        f1data = data[data['mag']<90]
        fdata = f1data[(f1data['sim_mag'] < high) & (f1data['sim_mag'] >= low)]
        #print len(fdata)
        if len(fdata) is 0:
            sqrtn[i_bin] = 0
            mean[i_bin] = 0  # np.median(fdata['mag'])
            stdv[i_bin] = 0  # np.std(fdata['mag'])
            continue
        sqrtn[i_bin] = np.sqrt(len(fdata['mag']))
        m, med, st = sigma_clipped_stats(fdata['mag'])
        mean[i_bin] = m  # np.mean(fdata['mag'])
        stdv[i_bin] = st  # np.std(fdata['mag'])
    if return_center_bins:
        return bins[:-1]+(high-low)*0.5, mean, stdv, sqrtn
    else:
        return mean, stdv, sqrtn, mean_sim_mag


def binning_res(data, bins, return_center_bins=False):
    mean = np.zeros_like(bins[:-1])
    stdv = np.zeros_like(bins[:-1])
    sqrtn= np.zeros_like(mean)
    mean_sim_mag = np.zeros_like(mean)
    for i_bin, low in enumerate(bins[:-1]):
        high = bins[i_bin+1]
        f1data = data[data['mag']<90]
        fdata = f1data[(f1data['sim_mag'] < high) & (f1data['sim_mag'] >= low)]
        fdata_mag = fdata['mag'] - fdata['sim_mag']
        if len(fdata) is 0:
            sqrtn[i_bin] = 0
            mean[i_bin] = np.mean(fdata_mag)
            stdv[i_bin] = np.std(fdata_mag)
            continue
        mean_sim_mag[i_bin] = (high+low)/2.
        sqrtn[i_bin] = np.sqrt(len(fdata_mag))
        mean[i_bin] = np.mean(fdata_mag)
        stdv[i_bin] = np.std(fdata_mag)
    if return_center_bins:
        return bins[:-1]+(high-low)*0.5, mean, stdv, sqrtn
    else:
        return mean, stdv, sqrtn, mean_sim_mag


def binning_res_robust(data, bins, return_center_bins=False):
    mean = np.zeros_like(bins[:-1])
    stdv = np.zeros_like(bins[:-1])
    sqrtn= np.zeros_like(mean)
    mean_sim_mag = np.zeros_like(mean)
    for i_bin, low in enumerate(bins[:-1]):
        high = bins[i_bin+1]
        f1data = data[data['mag']<90]
        fdata = f1data[(f1data['sim_mag'] < high) & (f1data['sim_mag'] >= low)]
        fdata_mag = fdata['mag'] - fdata['sim_mag']
        if len(fdata) is 0:
            sqrtn[i_bin] = 0
            mean[i_bin] = np.mean(fdata_mag)
            stdv[i_bin] = np.std(fdata_mag)
            continue
        mean_sim_mag[i_bin] = (high+low)/2.
        sqrtn[i_bin] = np.sqrt(len(fdata_mag))
        m, med, st = sigma_clipped_stats(fdata_mag)
        mean[i_bin] = m  # np.mean(fdata_mag)
        stdv[i_bin] = st  # np.std(fdata_mag)
    if return_center_bins:
        return bins[:-1]+(high-low)*0.5, mean, stdv, sqrtn
    else:
        return mean, stdv, sqrtn, mean_sim_mag


def custom_histogram(vector, bins=None, cumulative=False, errors=False):
    if bins is None:
        hh = np.histogram(vector)
    else:
        hh = np.histogram(vector, bins=bins)
    dx = hh[1][1] - hh[1][0]
    x_bins = hh[1][:-1] + dx

    if cumulative is True:
        vals = [sum(hh[0][:i+1]) for i, _ in enumerate(hh[0])]
        if errors:
            err = np.sqrt(hh[0])
            return x_bins, vals, err
        return x_bins, vals

    elif cumulative == -1:
        vals = [sum(hh[0][i:]) for i, _ in enumerate(hh[0])]
        if errors:
            err = np.sqrt(hh[0])
            return x_bins, vals, err
        return x_bins, vals

    else:
        if errors:
            err = np.sqrt(hh[0])
            return x_bins, hh[0], err

        return x_bins, hh[0]


from sklearn.linear_model import RANSACRegressor

def get_mags_iso(df):
    model = RANSACRegressor()
    try:
        model.fit(df['MAG_ISO'].values.reshape(-1, 1),
                  df['sim_mag'].values.reshape(-1, 1))
    except:
        mean_offset = sigma_clipped_stats(df['mag_offset_iso'])[0]
        slope = 1.0
        return [mean_offset, slope]
    mean_offset = model.estimator_.intercept_[0]
    slope = model.estimator_.coef_[0][0]

    res = [mean_offset, slope]
    return res


def cal_mags_iso(df):
    ids = []
    offsets = []
    slopes  = []
    for name, group in df.dropna().groupby(['image_id'], sort=False):
        b, a = get_mags_iso(group)
        ids.append(name)
        offsets.append(b)
        slopes.append(a)
    dd = pd.DataFrame(np.array([ids, offsets, slopes]).T,
                      columns=['image_id', 'mean_offset_iso', 'slope_iso'])
    return dd

def get_mags(df):
    model = RANSACRegressor()
    try:
        model.fit(df['MAG_APER'].values.reshape(-1, 1),
                  df['sim_mag'].values.reshape(-1, 1))
    except:
        mean_offset = sigma_clipped_stats(df['mag_offset'])[0]
        slope = 1.0
        mags = df['MAG_APER'] + mean_offset
        p05, p95 = np.percentile(mags, [5., 95.])
        return [mean_offset, slope, p05, p95]
    mean_offset = model.estimator_.intercept_[0]
    slope = model.estimator_.coef_[0][0]
    mask = model.inlier_mask_
    mags = model.predict(df['MAG_APER'].values.reshape(-1,1))
    p05, p95 = np.percentile(mags[mask], [5., 95.])

    res = [mean_offset, slope, p05, p95]
    return res


def cal_mags(df):
    ids = []
    offsets = []
    slopes  = []
    per_05 = []
    per_95 = []
    for name, group in df.dropna().groupby(['image_id'], sort=False):
        b, a, p05, p95 = get_mags(group)
        ids.append(name)
        offsets.append(b)
        slopes.append(a)
        per_05.append(p05)
        per_95.append(p95)

    dd = pd.DataFrame(np.array([ids, offsets, slopes, per_05, per_95]).T,
                      columns=['image_id', 'mean_offset', 'slope', 'p05', 'p95'])
    return dd

# =============================================================================
# Feature selectors
# =============================================================================
def importance_forest(X, y, forest=None, cols=None, method=None):
    if forest is None:
        forest = ExtraTreesClassifier(n_estimators=250,
                                      random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    #print indices
    #print importances
    #print cols
    # Print the feature ranking
    #print("Feature ranking:")

    # Plot the feature importances of the forest
    #plt.figure(figsize=(6, 6))
    plt.title("{}".format(method))
    plt.barh(range(X.shape[1])[0:8], importances[indices][0:8],
           color="r", xerr=std[indices][0:8], align="center")
    if cols is not None:
        plt.yticks(range(X.shape[1])[0:8], cols[indices-1][0:8], rotation='horizontal', fontsize=10)
    else:
        plt.yticks(range(X.shape[1]), indices)
    #plt.ylim([-1, X.shape[1]])
    plt.xlim(0, np.max(importances)+np.max(std))
    ax = plt.gca()
    ax.invert_yaxis()
    #plt.show()
    return [(cols[indices[f]-1], importances[indices[f]]) for f in range(X.shape[1])]

def full_importance_forest(X, y, forest=None, cols=None, method=None):
    if forest is None:
        forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    return [indices, importances, cols]

def importance_perm(X, y, forest=None, cols=None, method=None):

    X = pd.DataFrame(X, columns=cols)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if forest is None:
        forest = RandomForestClassifier(n_estimators=250, random_state=33, n_jobs=-1)

    X_train['Random'] = np.random.random(size=len(X_train))
    X_test['Random'] = np.random.random(size=len(X_test))

    forest.fit(X_train, y_train)
    imp = importances(forest, X_test, y_test) # permutation
    return imp


def importance_perm_kfold(X, y, forest=None, cols=None, method=None, nfolds=10):
    skf = StratifiedKFold(n_splits=nfolds)
    imp = []

    for train, test in skf.split(X, y):
        X_train = pd.DataFrame(X[train], columns=cols)
        X_test = pd.DataFrame(X[test], columns=cols)
        y_train = pd.DataFrame(y[train])
        y_test = pd.DataFrame(y[test])

        if forest is None:
            forest = RandomForestClassifier(n_estimators=250, random_state=33, n_jobs=-1)

        X_train['Random'] = np.random.random(size=len(X_train))
        X_test['Random'] = np.random.random(size=len(X_test))

        forest.fit(X_train, y_train)
        imp.append(importances(forest, X_test, y_test)) # permutation
    #imp = pd.concat(imp, axis=1)
    return imp

def select(X, Y, percentile, selector_f=mutual_info_classif, log=False):
    selector = SelectPercentile(selector_f, percentile=percentile)
    selector.fit(X, Y)
    scores = selector.scores_
    if log:
        scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    X_indices = np.arange(X.shape[-1]).reshape(1, -1)
    selected_cols = selector.transform(X_indices)
    return scores, selector, selected_cols

from rfpimp import importances
from rfpimp import plot_corr_heatmap

def importance_perm(X, y, forest=None, cols=None, method=None):

    X = pd.DataFrame(X, columns=cols)
    y = pd.DataFrame(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if forest is None:
        forest = RandomForestClassifier(n_estimators=250, random_state=33, n_jobs=-1)

    X_train['Random'] = np.random.random(size=len(X_train))
    X_test['Random'] = np.random.random(size=len(X_test))

    forest.fit(X_train, y_train)
    imp = importances(forest, X_test, y_test) # permutation
    return imp


def importance_perm_kfold(X, y, forest=None, cols=None, method=None, nfolds=10):
    skf = StratifiedKFold(n_splits=nfolds)
    imp = []

    for train, test in skf.split(X, y):
        X_train = pd.DataFrame(X[train], columns=cols)
        X_test = pd.DataFrame(X[test], columns=cols)
        y_train = pd.DataFrame(y[train])
        y_test = pd.DataFrame(y[test])

        if forest is None:
            forest = RandomForestClassifier(n_estimators=250, n_jobs=-1)

        X_train['Random'] = np.random.random(size=len(X_train))
        X_test['Random'] = np.random.random(size=len(X_test))

        forest.fit(X_train, y_train)
        imp.append(importances(forest, X_test, y_test)) # permutation
    return imp

# =============================================================================
# Experiment
# =============================================================================
def experiment(clf, x, y, nfolds=10, printing=False, probs=False,
               train_final=False):
    # import ipdb; ipdb.set_trace()
    skf = StratifiedKFold(n_splits=nfolds)
    probabilities = None # np.array([])
    predictions = np.array([])
    y_testing = np.array([])

    results = {}
    for train, test in skf.split(x, y):

        x_train = x[train]
        y_train = y[train]
        clf.fit(x_train, y_train)

        x_test = x[test]
        y_test = y[test]
        pr = clf.predict(x_test)
        if probs:
            probas = clf.predict_proba(x_test)  #[:, 0]

            probabilities = (
                probas if probabilities is None else
                np.vstack([probabilities, probas]))
        predictions = np.hstack([predictions, pr])
        y_testing = np.hstack([y_testing, y_test])

    results['y_test'] = y_testing
    results['predictions'] = predictions
    if probs:
        results['probabilities'] = probabilities

    if printing:
        print(metrics.classification_report(y_testing, predictions))

    if probs:
        fpr, tpr, thresholds = metrics.roc_curve(y_testing, 1.-probabilities[:, 0])
        prec_rec_curve = metrics.precision_recall_curve(y_testing, 1.- probabilities[:, 0])
        roc_auc = metrics.auc(fpr, tpr)

        results['fpr'] = fpr
        results['tpr'] = tpr
        results['thresh'] = thresholds
        results['roc_auc'] = roc_auc
        results['prec_rec_curve'] = prec_rec_curve

    if train_final:
        clf.fit(x, y)

    results['model'] = clf
    results['confusion_matrix'] = metrics.confusion_matrix(y_testing, predictions)
    results['bacc'] = metrics.balanced_accuracy_score(y_testing, predictions)
    results['acc'] = metrics.accuracy_score(y_testing, predictions)
    results['aprec'] = metrics.average_precision_score(y_testing, predictions)
    results['prec'] = metrics.precision_score(y_testing, predictions)
    results['reca'] = metrics.recall_score(y_testing, predictions)
    results['f1'] = metrics.f1_score(y_testing, predictions)

    return results


# =============================================================================
# funcion para ml
# =============================================================================
def group_ml(train_data, group_cols=['m1_diam', 'exp_time', 'new_fwhm'],
             target=['IS_REAL'], cols=['mag'], var_thresh=0.1, percentile=30.,
             method='Bramich'):
    rows = []
    i_group = 0
    for pars, data in train_data.groupby(group_cols):
        i_group += 1
        ## spliting the data into train and final test
        train, test = train_test_split(data[cols+target].dropna(), test_size=0.25,
                                       stratify=data[cols+target].dropna().IS_REAL)
        d = train[cols]
        y = train[target].values.ravel()

        scaler = preprocessing.StandardScaler().fit(d)
        X = scaler.transform(d)
        X_test = scaler.transform(test[cols])
        y_test = test.IS_REAL.values.ravel()

        # =====================================================================
        # building the rows of this gigantic table
        # =====================================================================
        # separate them in three groups
        row_knn = []
        row_rfo = []
        row_svc = []

        # =============================================================================
        # univariate cuts
        # =============================================================================
        thresh = var_thresh
        sel = VarianceThreshold(threshold=thresh)
        X = sel.fit_transform(X)
        X_test = sel.transform(X_test)
        newcols = d.columns[sel.get_support()]
        print('Dropped columns = {}'.format(d.columns[~sel.get_support()]))
        d = pd.DataFrame(X, columns=newcols)

        percentile = percentile
        scores, selector, selected_cols = select(X, y, percentile)
        scoring = pd.DataFrame(scores, index=newcols, columns=[method])
        selection = scoring.loc[newcols.values[selected_cols][0]]
        dat = pd.DataFrame(X, columns=newcols)[selection.index]

        # =============================================================================
        # KNN
        # =============================================================================
        model = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', n_jobs=-1)

        # experiment befor fslection
        rslt0_knn = experiment(model, X, y, printing=False, nfolds=5)
        row_knn.append(rslt0_knn['confusion_matrix'].flatten())
        row_knn.append(rslt0_knn['bacc'])
        row_knn.append(rslt0_knn['acc'])
        row_knn.append(rslt0_knn['aprec'])
        row_knn.append(rslt0_knn['prec'])
        row_knn.append(rslt0_knn['reca'])
        row_knn.append(rslt0_knn['f1'])

        # experiment after fselection
        rslt_knn = experiment(model, dat.values, y, printing=False, nfolds=5)
        row_knn.append(rslt_knn['confusion_matrix'].flatten())
        row_knn.append(rslt_knn['bacc'])
        row_knn.append(rslt_knn['acc'])
        row_knn.append(rslt_knn['aprec'])
        row_knn.append(rslt_knn['prec'])
        row_knn.append(rslt_knn['reca'])
        row_knn.append(rslt_knn['f1'])

        # test on the testset
        #  before fselection
        model.fit(X, y)
        preds = model.predict(X_test)
        test_cm_knn0 = metrics.confusion_matrix(y_test, preds)
        test_bacc_knn0 = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_knn0 = metrics.accuracy_score(y_test, preds)
        test_aprec_knn0 = metrics.average_precision_score(y_test, preds)
        test_prec_knn0 = metrics.precision_score(y_test, preds)
        test_reca_knn0 = metrics.recall_score(y_test, preds)
        test_f1_knn0 = metrics.f1_score(y_test, preds)

        row_knn += [test_cm_knn0.flatten(), test_bacc_knn0, test_acc_knn0,
                    test_aprec_knn0, test_prec_knn0, test_reca_knn0,
                    test_f1_knn0]

        #  after fselection
        model.fit(dat.values, y)
        preds = model.predict(selector.transform(X_test))
        test_cm_knn = metrics.confusion_matrix(y_test, preds)
        test_bacc_knn = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_knn = metrics.accuracy_score(y_test, preds)
        test_aprec_knn = metrics.average_precision_score(y_test, preds)
        test_prec_knn = metrics.precision_score(y_test, preds)
        test_reca_knn = metrics.recall_score(y_test, preds)
        test_f1_knn = metrics.f1_score(y_test, preds)

        row_knn += [test_cm_knn.flatten(), test_bacc_knn, test_acc_knn,
                    test_aprec_knn, test_prec_knn, test_reca_knn, test_f1_knn]
        # =============================================================================
        # randomforest
        # =============================================================================
        corr = d.corr()
        # remove corr columns
        correlated_features = set()
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.8:
                    colname = corr.columns[i]
                    correlated_features.add(colname)
        decorr = d.drop(correlated_features, axis=1)
        corr = decorr.corr()

        model = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        importance = importance_perm_kfold(decorr.values, y, model,
                                           cols=decorr.columns, method=method)

        res = pd.concat(importance, axis=1)
        full_cols = list(decorr.index).extend(['Random'])
        m = res.mean(axis=1).reindex(full_cols)
        s = res.std(axis=1).reindex(full_cols)

        thresh = m.loc['Random'] + 3*s.loc['Random']
        spikes = m - 3*s
        selected = spikes > thresh
        signif = (m - m.loc['Random'])/s
        selected = signif>2.5
        dat = d[selected[selected].index]

        n_fts = np.min([len(dat.columns), 7])
        model = RandomForestClassifier(n_estimators=800, max_features=n_fts,
                                       min_samples_leaf=20, n_jobs=-1)

        # experiment before fselection
        rslt0_rforest = experiment(model, X, y, printing=False, nfolds=5)
        row_rfo.append(rslt0_rforest['confusion_matrix'].flatten())
        row_rfo.append(rslt0_rforest['bacc'])
        row_rfo.append(rslt0_rforest['acc'])
        row_rfo.append(rslt0_rforest['aprec'])
        row_rfo.append(rslt0_rforest['prec'])
        row_rfo.append(rslt0_rforest['reca'])
        row_rfo.append(rslt0_rforest['f1'])

        # experiment after fselection
        rslt_rforest = experiment(model, dat.values, y, printing=False, nfolds=5)
        row_rfo.append(rslt_rforest['confusion_matrix'].flatten())
        row_rfo.append(rslt_rforest['bacc'])
        row_rfo.append(rslt_rforest['acc'])
        row_rfo.append(rslt_rforest['aprec'])
        row_rfo.append(rslt_rforest['prec'])
        row_rfo.append(rslt_rforest['reca'])
        row_rfo.append(rslt_rforest['f1'])

        d_test = pd.DataFrame(X_test, columns=newcols)[selected[selected].index]

        # test on the testset
        #  before fselection
        model.fit(X, y)
        preds = model.predict(X_test)
        test_cm_rforest0 = metrics.confusion_matrix(y_test, preds)
        test_bacc_rforest0 = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_rforest0 = metrics.accuracy_score(y_test, preds)
        test_aprec_rforest0 = metrics.average_precision_score(y_test, preds)
        test_prec_rforest0 = metrics.precision_score(y_test, preds)
        test_reca_rforest0 = metrics.recall_score(y_test, preds)
        test_f1_rforest0 = metrics.f1_score(y_test, preds)

        row_rfo += [test_cm_rforest0.flatten(), test_bacc_rforest0,
                    test_acc_rforest0, test_aprec_rforest0, test_prec_rforest0,
                    test_reca_rforest0, test_f1_rforest0]

        #  after fselection
        model.fit(dat.values, y)
        preds = model.predict(d_test.values)
        test_cm_rforest = metrics.confusion_matrix(y_test, preds)
        test_bacc_rforest = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_rforest = metrics.accuracy_score(y_test, preds)
        test_aprec_rforest = metrics.average_precision_score(y_test, preds)
        test_prec_rforest = metrics.precision_score(y_test, preds)
        test_reca_rforest = metrics.recall_score(y_test, preds)
        test_f1_rforest = metrics.f1_score(y_test, preds)

        row_rfo += [test_cm_rforest.flatten(), test_bacc_rforest,
                    test_acc_rforest, test_aprec_rforest, test_prec_rforest,
                    test_reca_rforest, test_f1_rforest]

        # =============================================================================
        # SVC
        # =============================================================================
        #svc = SVC(kernel='linear',
        #          cache_size=2048,
        #          class_weight='balanced',
        #          probability=True)
        svc = svm.LinearSVC(dual=False, tol=1e-5)
        rfecv = feature_selection.RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
                      scoring='accuracy', n_jobs=-1)

        rfecv.fit(np.ascontiguousarray(X), y)
        print("Optimal number of features : {}" .format(rfecv.n_features_))
        sel_cols = newcols[rfecv.support_]
        print(sel_cols)
        dat = d[sel_cols]

        model = svc
        # experiment before fselection
        rslt0_svc = experiment(model, X, y, printing=False, nfolds=5)
        row_svc.append(rslt0_svc['confusion_matrix'].flatten())
        row_svc.append(rslt0_svc['bacc'])
        row_svc.append(rslt0_svc['acc'])
        row_svc.append(rslt0_svc['aprec'])
        row_svc.append(rslt0_svc['prec'])
        row_svc.append(rslt0_svc['reca'])
        row_svc.append(rslt0_svc['f1'])

        # experiment after fselection
        rslt_svc = experiment(model, dat.values, y, printing=False, nfolds=5)
        row_svc.append(rslt_svc['confusion_matrix'].flatten())
        row_svc.append(rslt_svc['bacc'])
        row_svc.append(rslt_svc['acc'])
        row_svc.append(rslt_svc['aprec'])
        row_svc.append(rslt_svc['prec'])
        row_svc.append(rslt_svc['reca'])
        row_svc.append(rslt_svc['f1'])

        d_test = pd.DataFrame(X_test, columns=newcols)[sel_cols].values

        # test on the testset
        #  before fselection
        model.fit(X, y)
        preds = model.predict(X_test)
        test_acc_svc0 = metrics.accuracy_score(y_test, preds)
        test_cm_svc0 = metrics.confusion_matrix(y_test, preds)
        test_bacc_svc0 = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_svc0 = metrics.accuracy_score(y_test, preds)
        test_aprec_svc0 = metrics.average_precision_score(y_test, preds)
        test_prec_svc0 = metrics.precision_score(y_test, preds)
        test_reca_svc0 = metrics.recall_score(y_test, preds)
        test_f1_svc0 = metrics.f1_score(y_test, preds)

        row_svc += [test_cm_svc0.flatten(), test_bacc_svc0, test_acc_svc0,
                    test_aprec_svc0, test_prec_svc0, test_reca_svc0,
                    test_f1_svc0]

        #  after fselection
        model.fit(dat.values, y)
        preds = model.predict(d_test)
        test_acc_svc = metrics.accuracy_score(y_test, preds)
        test_cm_svc = metrics.confusion_matrix(y_test, preds)
        test_bacc_svc = metrics.balanced_accuracy_score(y_test, preds)
        test_acc_svc = metrics.accuracy_score(y_test, preds)
        test_aprec_svc = metrics.average_precision_score(y_test, preds)
        test_prec_svc = metrics.precision_score(y_test, preds)
        test_reca_svc = metrics.recall_score(y_test, preds)
        test_f1_svc = metrics.f1_score(y_test, preds)

        row_svc += [test_cm_svc.flatten(), test_bacc_svc, test_acc_svc,
                    test_aprec_svc, test_prec_svc, test_reca_svc,
                    test_f1_svc]


        vals = pars + row_knn + row_rfo + row_svc
        rows.append(np.array(vals).flatten())
        print('{} groups processed'.format(i_group))

    knn_cols = ['knn_exp0_c00', 'knn_exp0_c01', 'knn_exp0_c10', 'knn_exp0_c11',
                'knn_exp0_bacc', 'knn_exp0_acc', 'knn_exp0_prec',
                'knn_exp0_reca', 'knn_exp0_f1',
                'knn_exp_c00', 'knn_exp_c01', 'knn_exp_c10', 'knn_exp_c11',
                'knn_exp_bacc', 'knn_exp_acc', 'knn_exp_prec',
                'knn_exp_reca', 'knn_exp_f1',
                'knn_test0_c00', 'knn_test0_c01', 'knn_test0_c10', 'knn_test0_c11',
                'knn_test0_bacc', 'knn_test0_acc', 'knn_test0_prec',
                'knn_test0_reca', 'knn_test0_f1',
                'knn_test_c00', 'knn_test_c01', 'knn_test_c10', 'knn_test_c11',
                'knn_test_bacc', 'knn_test_acc', 'knn_test_prec',
                'knn_test_reca', 'knn_test_f1']

    rfo_cols = ['rfo_exp0_c00', 'rfo_exp0_c01', 'rfo_exp0_c10', 'rfo_exp0_c11',
                'rfo_exp0_bacc', 'rfo_exp0_acc', 'rfo_exp0_prec',
                'rfo_exp0_reca', 'rfo_exp0_f1',
                'rfo_exp_c00', 'rfo_exp_c01', 'rfo_exp_c10', 'rfo_exp_c11',
                'rfo_exp_bacc', 'rfo_exp_acc', 'rfo_exp_prec',
                'rfo_exp_reca', 'rfo_exp_f1',
                'rfo_test0_c00', 'rfo_test0_c01', 'rfo_test0_c10', 'rfo_test0_c11',
                'rfo_test0_bacc', 'rfo_test0_acc', 'rfo_test0_prec',
                'rfo_test0_reca', 'rfo_test0_f1',
                'rfo_test_c00', 'rfo_test_c01', 'rfo_test_c10', 'rfo_test_c11',
                'rfo_test_bacc', 'rfo_test_acc', 'rfo_test_prec',
                'rfo_test_reca', 'rfo_test_f1']

    svc_cols = ['svc_exp0_c00', 'svc_exp0_c01', 'svc_exp0_c10', 'svc_exp0_c11',
                'svc_exp0_bacc', 'svc_exp0_acc', 'svc_exp0_prec',
                'svc_exp0_reca', 'svc_exp0_f1',
                'svc_exp_c00', 'svc_exp_c01', 'svc_exp_c10', 'svc_exp_c11',
                'svc_exp_bacc', 'svc_exp_acc', 'svc_exp_prec',
                'svc_exp_reca', 'svc_exp_f1',
                'svc_test0_c00', 'svc_test0_c01', 'svc_test0_c10', 'svc_test0_c11',
                'svc_test0_bacc', 'svc_test0_acc', 'svc_test0_prec',
                'svc_test0_reca', 'svc_test0_f1',
                'svc_test_c00', 'svc_test_c01', 'svc_test_c10', 'svc_test_c11',
                'svc_test_bacc', 'svc_test_acc', 'svc_test_prec',
                'svc_test_reca', 'svc_test_f1']
    ml_cols = group_cols + knn_cols + rfo_cols + svc_cols
    ml_results = pd.DataFrame(rows, columns=ml_cols)
    return ml_results



transl = {u'thresh': u'THRESHOLD',
          u'peak': u'FLUX_MAX',
          u'x': u'X_IMAGE',
          u'y': u'Y_IMAGE',
          u'x2': u'X2_IMAGE',
          u'y2': u'Y2_IMAGE',
          u'xy': u'XY_IMAGE',
          u'a':u'A_IMAGE',
          u'b':u'B_IMAGE',
          u'theta':u'THETA_IMAGE',
          u'cxx':u'CXX_IMAGE',
          u'cyy':u'CYY_IMAGE',
          u'cxy':u'CXY_IMAGE',
          u'cflux':u'FLUX_ISO',
          u'flux':u'FLUX_APER',
          u'flag': u'FLAGS',
          u'DELTAX': u'DELTAX',
          u'DELTAY': u'DELTAY',
          u'RATIO': u'RATIO',
          u'ROUNDNESS': u'ROUNDNESS',
          u'PEAK_CENTROID': u'PEAK_CENTROID',
          #u'MAG': u'mag',
          u'MU': u'MU',
          u'SN': u'SN'}

detransl = {v: k for k, v in transl.items()}

