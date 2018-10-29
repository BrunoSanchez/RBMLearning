
import numpy as np
import pandas as pd

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
