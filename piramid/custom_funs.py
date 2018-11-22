
import numpy as np
import pandas as pd

from astropy.stats import sigma_clipped_stats

from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

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


def experiment(clf, x, y, nfolds=10, printing=False, probs=True):
    skf = StratifiedKFold(n_splits=nfolds)
    probabilities = None # np.array([])
    predictions = np.array([])
    y_testing = np.array([])

    for train, test in skf.split(x, y):

        x_train = x[train]
        y_train = y[train]
        clf.fit(x_train, y_train)

        x_test = x[test]
        y_test = y[test]
        pr = clf.predict(x_test)

        probs = clf.predict_proba(x_test)  #[:, 0]

        probabilities = (
            probs if probabilities is None else
            np.vstack([probabilities, probs]))
        predictions = np.hstack([predictions, pr])
        y_testing = np.hstack([y_testing, y_test])

    if printing:
        print(metrics.classification_report(y_testing, predictions))
    fpr, tpr, thresholds = metrics.roc_curve(y_testing, 1.-probabilities[:, 0])
    prec_rec_curve = metrics.precision_recall_curve(y_testing, 1.- probabilities[:, 0])
    roc_auc = metrics.auc(fpr, tpr)
    clf.fit(x, y)
    return {'fpr': fpr,
            'tpr': tpr,
            'thresh': thresholds,
            'roc_auc': roc_auc,
            'prec_rec_curve': prec_rec_curve,
            'y_test': y_testing,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': metrics.confusion_matrix(y_testing, predictions),
            'model' : clf
            }


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

