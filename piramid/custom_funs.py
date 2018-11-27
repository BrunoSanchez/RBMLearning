
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
        fdata = f1data[(f1data['sim_mag'] < high) * (f1data['sim_mag'] >= low)]
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
        fdata = f1data[(f1data['sim_mag'] < high) * (f1data['sim_mag'] >= low)]
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
        fdata = f1data[(f1data['sim_mag'] < high) * (f1data['sim_mag'] >= low)]
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
        fdata = f1data[(f1data['sim_mag'] < high) * (f1data['sim_mag'] >= low)]
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


def experiment(clf, x, y, nfolds=10, printing=False, probs=True,
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
            probs = clf.predict_proba(x_test)  #[:, 0]

            probabilities = (
                probs if probabilities is None else
                np.vstack([probabilities, probs]))
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

    return results


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

from sklearn.ensemble import ExtraTreesClassifier

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
# funcion para ml
# =============================================================================
def group_ml(train_data, group_cols=['m1_diam', 'exp_time', 'new_fwhm'],
             target=['IS_REAL'], cols=['mag'], var_thresh=0.1, percentile=30.,
             method='Bramich'):
    rows = []
    for pars, data in train_data.groupby(group_cols):
        train, test = train_test_split(data[cols+target].dropna(), test_size=0.25,
                                       stratify=data[cols+target].dropna().IS_REAL)
        d_ois = train[cols]
        y_ois = train[target]

        scaler = preprocessing.StandardScaler().fit(d_ois)
        X_ois = scaler.transform(d_ois)
        X_test_ois = scaler.transform(test[cols])

        # =============================================================================
        # univariate
        # =============================================================================
        thresh = var_thresh
        sel = VarianceThreshold(threshold=thresh)
        X_ois = sel.fit_transform(X_ois)
        X_test_ois = sel.transform(X_test_ois)
        newcols_ois = d_ois.columns[sel.get_support()]
        print('Dropped columns = {}'.format(d_ois.columns[~sel.get_support()]))
        d_ois = pd.DataFrame(X_ois, columns=newcols_ois)

        percentile = percentile
        scores, selector, selected_cols = select(X_ois, y_ois, percentile)
        scoring_ois = pd.DataFrame(scores, index=newcols_ois, columns=['ois'])
        selection_ois = scoring_ois.loc[newcols_ois.values[selected_cols][0]]
        dat_ois = pd.DataFrame(X_ois, columns=newcols_ois)[selection_ois.index]

        # =============================================================================
        # KNN
        # =============================================================================
        model = neighbors.KNeighborsClassifier(n_neighbors=7, weights='uniform', n_jobs=-1)

        rslt0_knn_ois_uniform = experiment(model, X_ois, y_ois.values.ravel(), printing=True)
        model.fit(X_ois, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        rslt0_knn_ois_uniform['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_knn0 = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        rslt0_knn_ois_uniform['test_bacc'] = acc_knn0

        rslts_knn_ois_uniform = experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(dat_ois.values, y_ois.values.ravel())
        preds = model.predict(selector.transform(X_test_ois))
        rslts_knn_ois_uniform['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_knn = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        rslt0_knn_ois_uniform['test_bacc'] = acc_knn

        # =============================================================================
        # randomforest
        # =============================================================================
        corr = d_ois.corr()
        # Generate a mask for the upper triangle
        mask = np.zeros_like(corr, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        # remove corr columns
        correlated_features = set()
        for i in range(len(corr.columns)):
            for j in range(i):
                if abs(corr.iloc[i, j]) > 0.8:
                    colname = corr.columns[i]
                    correlated_features.add(colname)
        decorr_ois = d_ois.drop(correlated_features, axis=1)
        corr = decorr_ois.corr()

        model = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)
        ois_importance = importance_perm_kfold(decorr_ois.values, y_ois.values.ravel(),
            model, cols=decorr_ois.columns, method=method)

        res_ois = pd.concat(ois_importance, axis=1)
        full_cols = list(decorr_ois.index).extend(['Random'])
        m = res_ois.mean(axis=1).reindex(full_cols)
        s = res_ois.std(axis=1).reindex(full_cols)

        thresh = m.loc['Random'] + 3*s.loc['Random']
        spikes = m - 3*s
        selected = spikes > thresh
        signif = (m - m.loc['Random'])/s
        selected = signif>2.5
        dat_ois = d_ois[selected[selected].index]

        n_fts = np.min([len(dat_ois.columns), 7])
        model = RandomForestClassifier(n_estimators=800, max_features=n_fts,
                                       min_samples_leaf=20, n_jobs=-1)
        rslts_ois_rforest = experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(dat_ois.values, y_ois.values.ravel())
        d_test = pd.DataFrame(X_test_ois, columns=newcols_ois)[selected[selected].index]
        preds = model.predict(d_test.values)
        #rslts_ois_rforest['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_rforest = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslts_ois_rforest['test_bacc'] = acc_rforest

        rslt0_ois_rforest = experiment(model, d_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(d_ois.values, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        # rslt0_ois_rforest['test_preds'] = preds
        print(metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_rforest0 = metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        # rslt0_ois_rforest['test_bacc'] = acc_rforest0

# =============================================================================
# SVC
# =============================================================================
        #svc = SVC(kernel='linear',
        #          cache_size=2048,
        #          class_weight='balanced',
        #          probability=True)
        #svc = svm.LinearSVC(dual=False, tol=1e-5)
        rfecv = feature_selection.RFECV(estimator=svc, step=1, cv=StratifiedKFold(6),
                      scoring='accuracy', n_jobs=-1)

        rfecv.fit(np.ascontiguousarray(X_ois), y_ois.values.ravel())
        print("Optimal number of features : {}" .format(rfecv.n_features_))
        sel_cols_ois = newcols_ois[rfecv.support_]
        print(sel_cols_ois)
        dat_ois = d_ois[sel_cols_ois]

        model = svc
        rslts_ois_svc = cf.experiment(model, dat_ois.values, y_ois.values.ravel(), printing=True, probs=False)
        model.fit(dat_ois.values, y_ois.values.ravel())
        preds = model.predict(pd.DataFrame(X_test_ois, columns=newcols_ois)[sel_cols_ois].values)
        #rslts_ois_svc['test_preds'] = preds
        print(cf.metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_svc = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslts_ois_svc['test_bacc'] = acc_svc

        rslt0_ois_svc = cf.experiment(model, d_ois.values, y_ois.values.ravel(), printing=True)
        model.fit(d_ois.values, y_ois.values.ravel())
        preds = model.predict(X_test_ois)
        #rslt0_ois_svc['test_preds'] = preds
        print(cf.metrics.classification_report(test.IS_REAL.values.ravel(), preds))
        acc_svc0 = cf.metrics.accuracy_score(test.IS_REAL.values.ravel(), preds)
        #rslt0_ois_svc['test_bacc'] = acc_svc0


        vals = [acc_knn0, acc_knn, acc_rforest0, acc_rforest, acc_svc, acc_svc0]
        rows.append(list(pars)+vals)

    ml_cols = ['m1_diam', 'exp_time', 'new_fwhm', 'acc_knn0', 'acc_knn',
               'acc_rforest0', 'acc_rforest', 'acc_svc', 'acc_svc0']
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

