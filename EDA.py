import polars as pl
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from scipy.stats import kurtosis as kurtosis_scipy
from scipy.stats import skew
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler 
import random
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations, product
import matplotlib.patches as mpatches
from collections import Counter


######################################################################################################################

def dtypes_df(df):
    
    """
    Parameters (inputs)
    ----------
    df: a Polars data-frame.     

    Returns (outputs)
    -------
    Python_types_df: a Polars data-frame with the Python type of each column (variable) of df data-frame.
    """

    Python_types_df = pl.DataFrame({'Columns' : df.columns, 'Python_type' : df.dtypes})

    return Python_types_df


def change_type(df, col_name, new_type) :

    """
    Parameters (inputs)
    ----------
    df: a Polars data-frame.    
    col: the name of a column from df. 
    new_type: the name of the Python type that you want to set for col_name. 

    Returns (outputs)
    -------
    The function changes the type of the col_name column of the df data-frame to new_type.
    """

    return df.with_columns(df[col_name].cast(new_type))    

######################################################################################################################

def columns_names(df, types=[pl.Float64, pl.Int64]):

    columns_names_ = [col for col in df.columns 
                            if df.select(pl.col(col)).dtypes[0] in types]
    
    return columns_names_

######################################################################################################################

def weighted_mean(X,w):

    """
    Parameters (inputs)
    ----------
    X: a Pandas series (the statistical variable).    
    w: an 1-D Numpy array with the weights (the weights vector). 
    
    Returns (outputs)
    -------
    weighted_mean_: the weighted mean of X using w as weights vector, computed using the formula presented above.
    """
    
    if not isinstance(X, np.ndarray) :
        X = X.to_numpy()

    weighted_mean_ = (1/np.sum(w))*np.sum(X*w)

    return weighted_mean_


def inv_quantile(X, h, scipy=False):

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).   
    h: the order of the inverse quantile. Must be a number in the domain of the variable represented by X. 
        
    Returns (outputs)
    -------
    inv_quantile_: the h-order inverse quantile of X.
    """

    if scipy == False:

        inv_quantile_ = round((X <= h).sum() / len(X), 4)

    else:

        inv_quantile_ = round(percentileofscore(X, h), 4)

    return inv_quantile_


def kurtosis(X, scipy=False):

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).    
        
    Returns (outputs)
    -------
    kurtosis_: the kurtosis of X.
    """
    if scipy == False:

        n = len(X)
        X_mean = X.mean()
        X_std = X.std()
        mu4 = (1/n)*((X - X_mean)**4).sum()
        kurtosis_ = mu4/(X_std**4) 

    else:

        kurtosis_ = kurtosis_scipy(X, fisher=False)

    return kurtosis_


def skewness(X, scipy=False):

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).    
        
    Returns (outputs)
    -------
    skewness_: the skewness of X.
    """

    if scipy == False :

        n = len(X)
        X_mean = X.mean()
        X_std = X.std()
        mu3 = (1/n)*((X - X_mean)**3).sum()
        skewness_ = (mu3/(X_std**3))

    else:

        skewness_ = skew(X)

    return skewness_


def MAD(X):

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).    
        
    Returns (outputs)
    -------
    MAD_: the median absolute deviation of X.
    """

    X_median = X.median()
    MAD_ = ((X- X_median).abs()).median()

    return MAD_

######################################################################################################################

def outlier_detection(X, h=1.5) :

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).
    h: a real number >= 0.
        
    Returns (outputs)
    -------
    trimmed_variable: the trimmed X variable, that is, a polars series with the values of X that are not outliers.
    outliers: a polars series with the outliers of X variable.
    """

    if isinstance(X, pl.DataFrame):
        # Transform a pl.DataFrame as pl.Series
        X = X[X.columns[0]]

    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + h*IQR
    lower_bound = Q1 - h*IQR

    trimmed_variable = X.filter((X >= lower_bound) & (X <= upper_bound))
    outliers = X.filter((X < lower_bound) | (X > upper_bound))  

    return trimmed_variable , outliers, lower_bound, upper_bound


def outliers_table(df, auto=True, col_names=[], h=1.5) :

    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data matrix).
    auto: can be True or False. If True, quantitative columns are detect automatically. If false, the function use the columns named in cols_list.
    cols_list: a list with the names of the columns that will be used in the case that auto=False.    
    h: a real number >= 0.

    Returns (outputs)
    -------
    df_outliers: a polars data-frame with the number of outliers/not outliers and the proportion of them, for the quantitative variables of df (if auto=True) 
                 or for the  variables of df whom names are in cols_list (if auto=False).
    """

    n_outliers = []
    n_not_outliers = []

    if auto == True : 

        quant_col_names = columns_names(df, types = [pl.Float64, pl.Int64])

    elif auto == False : 

        quant_col_names = col_names
     
    lower_bound_list, upper_bound_list = [], []

    for col in quant_col_names :
        
        trimmed_variable , outliers, lower_bound, upper_bound = outlier_detection(X=df[col], h=h)
        n_outliers.append(len(outliers))
        n_not_outliers.append(len(trimmed_variable))
        lower_bound_list.append(lower_bound)
        upper_bound_list.append(upper_bound)

    df_outliers = pl.DataFrame({'quant_variables': quant_col_names, 'lower_bound': lower_bound_list, 
                                'upper_bound': upper_bound_list, 'n_outliers': n_outliers, 'n_not_outliers': n_not_outliers})
    df_outliers = df_outliers.with_columns([(pl.col('n_outliers') / (pl.col('n_outliers') + pl.col('n_not_outliers'))).alias('prop_outliers')])
    df_outliers = df_outliers.with_columns([(1 - pl.col('prop_outliers')).alias('prop_not_outliers')])

    return df_outliers 


def outlier_filter(df, col, h=1.5) :

    """
    Parameters (inputs)
    ----------
    X: a Polars series (the statistical variable).
    h: a real number >= 0.
        
    Returns (outputs)
    -------
    trimmed_variable: the trimmed X variable, that is, a polars series with the values of X that are not outliers.
    outliers: a polars series with the outliers of X variable.
    """

    Q3 = df[col].quantile(0.75)
    Q1 = df[col].quantile(0.25)
    IQR = Q3 - Q1
    upper_bound = Q3 + h*IQR
    lower_bound = Q1 - h*IQR
    df_trimmed = df.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))

    return df_trimmed

######################################################################################################################

def quant_to_cat(X, rule, t=0.05, n_intervals=20, random_seed=123, custom_bins=None):

    """
    Parameters (inputs)
    ----------
    X: a polars series or a numpy array (the quantitative variable).
    rule: the name of the categorization rule to be used. The allowed names are 'default', 'mean', 'median', 'quartiles', 'deciles', 'quantiles', 'Scott', 'random'.
    t: a real number between 0 and 1. 
    n_intervals: the number of intervals taken into account when rule='default' or rule='random'.
    intervals: is a boolean, so takes True or False.
               If True, the categorization intervals will be return. If False, not.
    custom_bins: a list with the bins of the custom intervals. Will be used if rule='custom intervals'.
 
    Returns (outputs)
    -------
    X_cat: the categorical version of X.
    intervals: the categorization intervals. Only return if intervals=True.
    """ 
    if isinstance(X, np.ndarray):
        X = pl.Series(X)


    if rule == 'custom_intervals' :

        X_cat = X.cut(breaks=custom_bins)
       
    elif rule == 'mean':  

        X_min = X.min()
        X_max = X.max()
        X_mean = round(X.mean(), 4)
        eps = (X.max() - X.min()) * 0.01
        intervals_limits = [X_min-eps, X_mean, X_max]
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'median':

        X_min = X.min()
        X_max = X.max()
        X_median = round(X.median(), 4)
        eps = (X.max() - X.min()) * 0.01
        intervals_limits = [X_min - eps, X_median, X_max]
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'quartiles':

        X_min = X.min()
        X_max = X.max()
        Q25 = round(X.quantile(0.25), 4)
        Q50 = round(X.quantile(0.50), 4)
        Q75 = round(X.quantile(0.75), 4)
        eps = (X.max() - X.min()) * 0.01
        intervals_limits = [X_min - eps, Q25, Q50, Q75, X_max]
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'deciles':

        eps = (X.max() - X.min()) * 0.01
        intervals_limits = []
        for q in np.arange(0, 1.1, step=0.1) :
            Q = round(X.quantile(q), 4)
            intervals_limits.append(Q)
        intervals_limits[0] = intervals_limits[0] - eps
        intervals_limits.append(round(X.quantile(1), 4))
        intervals_limits = list(set(intervals_limits))                        
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'quantiles':
         
        eps = (X.max() - X.min()) * 0.01
        intervals_limits = []
        for q in np.arange(0, 1, step=t) :
            Q = round(X.quantile(q), 4)
            intervals_limits.append(Q)
        intervals_limits[0] = intervals_limits[0] - eps
        intervals_limits.append(round(X.quantile(1), 4))
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'Scott': # A modification of the Scott's rule.

        X_min = X.min()
        X_max = X.max()
        eps = (X.max() - X.min()) * 0.01       

        def scott_intervals(h) :
            w = np.ceil((X_max - X_min)/h)
            L = [None for x in range(h)]
            L[0] = X_min - eps
            for i in range(1, h):
                if L[i-1] < X_max:
                    L[i] = L[0] + i*w
                else:
                    break
            L = np.array(L)
            if np.any(L == None):
                L = list(L)
                L = [x for x in L if x != None]
            if not np.any(np.array(L) >= X_max):
                L = list(L)
                L.append(X_max)                
            return L
        
        intervals_limits = scott_intervals(h=n_intervals)
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    elif rule == 'random':

        X_min = X.min()
        X_max = X.max()
        eps = (X.max() - X.min()) * 0.01       

        def random_intervals(X=X, n_intervals=n_intervals, random_seed=random_seed) :
            random.seed(random_seed)
            L_inner = random.sample(range(int(X_min), int(X_max)+1), n_intervals-1)
            L = [X_min - eps] + L_inner + [X_max]
            L.sort()             
            return L
        
        intervals_limits = random_intervals(X, n_intervals)
        intervals_limits = list(set(intervals_limits))  
        X_cat = X.cut(breaks=intervals_limits)

    
    return X_cat
    
######################################################################################################################

def freq_table(X, intervals=None) :

    # Function to extract the lower bound from each interval
    def extract_lower_bound(interval):
        if interval.startswith("(-inf"):
            return float('-inf')
        else:
            return float(interval.split(", ")[0][1:])

    if intervals == None:
        unique_values, counts = np.unique(X, return_counts=True)
        unique_values_sorted_indices = np.argsort(unique_values)
        sorted_intervals = unique_values[unique_values_sorted_indices]
        counts = counts[unique_values_sorted_indices]
        rel_counts = counts / len(X)

    if intervals != None :
        unique_values, counts = np.unique(X, return_counts=True)
        lower_bounds = np.array([extract_lower_bound(interval) for interval in unique_values])
        unique_values_sorted_indices = np.argsort(lower_bounds)
        sorted_intervals = unique_values[unique_values_sorted_indices]
        counts = counts[unique_values_sorted_indices]
        rel_counts = counts / len(X)
        intervals_dict = dict()
        for x, int in enumerate(intervals):
            intervals_dict[int] = x
        intervals_not_null = [int for int in intervals if intervals_dict[int] in sorted_intervals]
        unique_values = intervals_not_null


    if isinstance(X, (np.ndarray, list)):

        title = 'unique values'

    else :

        title = X.name + ': unique values'

    freq_df = pl.DataFrame({title : sorted_intervals, 
                                'abs_freq' : counts, 
                                'rel_freq' : np.round(rel_counts, 4), 
                                'cum_abs_freq' : np.cumsum(counts), 
                                'cum_rel_freq' : np.cumsum(rel_counts)})
         

    return freq_df

######################################################################################################################

def standard_encoding(X) :

    X_std_encoding = OrdinalEncoder(dtype=int).fit_transform(X)
    X_std_encoding = X_std_encoding.flatten()
    
    return  X_std_encoding

######################################################################################################################

def standard_scaling(X) :
   
   """
   Parameters (inputs)
   ----------
   X: a polars series or a numpy array (the variable).

   Returns (outputs)
   -------
   scaler_: a numpy array with the standard scaling version of X.
   """

   scaler = StandardScaler().fit(X)
   scaler_ = scaler.transform(X)

   return scaler_

######################################################################################################################

def normalization(X, a, b) :   

   """
   Parameters (inputs)
   ----------
   X: a polars series or a numpy array (the variable).
   a,b : real numbers such that a <= b.
   
   Returns (outputs)
   -------
   min_max_scaler_: a numpy array with the (a,b) normalization version of X.
   """

   min_max_scaler = MinMaxScaler(feature_range=(a,b))
   min_max_scaler_ = min_max_scaler.fit_transform(X)
    
   return min_max_scaler_

######################################################################################################################

def transform_to_dummies(X, cols_to_dummies, drop_first=True):

    if isinstance(X, pd.DataFrame):
        X = pl.from_pandas(X)
    elif isinstance(X, np.ndarray):
        X = pl.from_numpy(X)

    for col in cols_to_dummies : 
        if drop_first == True:
            X = X.with_columns(X[[col]].to_dummies()[:,1:])
        else:
            X = X.with_columns(X[[col]].to_dummies())

    X = X[[col for col in X.columns if col not in cols_to_dummies]]

    return X

######################################################################################################################

def count_cols_nulls(X):
    return X.null_count()

######################################################################################################################

def prop_cols_nulls(X):
    num_rows = len(X)
    return  count_cols_nulls(X) / num_rows

######################################################################################################################

def count_row_nulls(X):

    X = X.to_numpy()
    null_count = np.sum(np.isnan(X), axis=1)
    
    if len(X) > 1 :
        return null_count
    elif len(X) == 1:
        return null_count[0]
    
######################################################################################################################

def prop_row_nulls(X) :

    num_columns = X.shape[1]
    return count_row_nulls(X) / num_columns

######################################################################################################################

def all_null_colum(X):
    return count_cols_nulls(X) == len(X)

######################################################################################################################

def any_null_colum(X):
    return count_cols_nulls(X) > 0

######################################################################################################################

def too_much_nulls_colums(X, limit):
    return prop_cols_nulls(X) > limit

######################################################################################################################

def null_imputation(df, auto_col=True, quant_col_names=None, cat_col_names=None, quant_method='mean', cat_method='mode', quant_value=None, cat_value=None) :

    """
    Parameters (inputs)
    ----------
    df: a pandas data-frame (the data-matrix).
    quant_method: the name of the method that will be used for quantitative data NaN imputation.
 
    Returns (outputs)
    ----------
    df_new: A pandas data-frame based on df but with all the df missing values filled or imputed.
    """

    # TO ADD: KNN IMPUTER

    df_new = df
    if auto_col == True :
        quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])
        cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])
    
    if quant_col_names is not None :

        for col in quant_col_names :

            if quant_method == 'mean' : 
                
               mean = df_new[col].mean()
               df_new = df_new.with_columns(pl.col(col).fill_null(mean))

            elif quant_method == 'median' : 
                
               median = df_new[col].median()
               df_new = df_new.with_columns(pl.col(col).fill_null(median))

            elif quant_method == 'Q25':

               Q25 = df_new[col].quantile(0.25)
               df_new = df_new.with_columns(pl.col(col).fill_null(Q25))            

            elif quant_method == 'Q75':

               Q75 = df_new[col].quantile(0.75)
               df_new = df_new.with_columns(pl.col(col).fill_null(Q75))  

            elif quant_method == 'max':

               max = df_new[col].max()
               df_new = df_new.with_columns(pl.col(col).fill_null(max)) 

            elif quant_method == 'min':

               min = df_new[col].min()
               df_new = df_new.with_columns(pl.col(col).fill_null(min)) 

            elif quant_method == 'free':

               df_new = df_new.with_columns(pl.col(col).fill_null(quant_value))

    if cat_col_names is not None :
        
        for col in cat_col_names :
            
            if cat_method == 'mode':
            
               mode = df_new[col].mode()
               df_new = df_new.with_columns(pl.col(col).fill_null(mode))

            elif cat_method == 'free':

               df_new = df_new.with_columns(pl.col(col).fill_null(cat_value))

    return df_new

######################################################################################################################

def summary(df, auto_col=True, quant_col_names=[], cat_col_names=[]) :

    if auto_col == True :
        quant_col_names = columns_names(df, types=[pl.Float64, pl.Int64])
        cat_col_names = columns_names(df, types=[pl.Boolean, pl.Utf8])

    elif auto_col == False : 
        # colnames debe ser un objeto tipo colnames = [list_quant_columns , list_cat_columns]
        # Ejemplo: colnames = [['quant_1' , 'quant_2'], ['cat_1', 'cat_2', 'cat_3']]
        quant_col_names = quant_col_names
        cat_col_names = cat_col_names

    n_rows = len(df)

    if len(quant_col_names) > 0 :
         
        # mean_quant_cols = [df[col].mean() for col in quant_col_names]
        mean_quant_cols = df[quant_col_names].mean().to_numpy().flatten()
        # std_quant_cols = [df[col].std() for col in quant_col_names]
        std_quant_cols = df[quant_col_names].std().to_numpy().flatten()
        # median_quant_cols = [df[col].median() for col in quant_col_names]
        median_quant_cols = df[quant_col_names].median().to_numpy().flatten()
        # Q25_quant_cols = [df[col].quantile(0.10) for col in quant_col_names]
        Q25_quant_cols = df[quant_col_names].quantile(0.25).to_numpy().flatten()
        # Q10_quant_cols = [df[col].quantile(0.25) for col in quant_col_names]
        Q10_quant_cols = df[quant_col_names].quantile(0.10).to_numpy().flatten()
        # Q75_quant_cols = [df[col].quantile(0.75) for col in quant_col_names]
        Q75_quant_cols = df[quant_col_names].quantile(0.75).to_numpy().flatten()
        # Q90_quant_cols = [df[col].quantile(0.90) for col in quant_col_names]
        Q90_quant_cols = df[quant_col_names].quantile(0.90).to_numpy().flatten()
        # max_quant_cols = [df[col].max() for col in quant_col_names]
        max_quant_cols = df[quant_col_names].max().to_numpy().flatten()
        # min_quant_cols = [df[col].min() for col in quant_col_names]
        min_quant_cols = df[quant_col_names].min().to_numpy().flatten()
        kurtosis_quant_cols = [kurtosis(df[col], scipy=True) for col in quant_col_names]
        skewness_quant_cols = [skewness(df[col], scipy=True) for col in quant_col_names]

        n_outliers, prop_outliers = [], []
        for col in quant_col_names :
           trimmed_variable, outliers, lower_bound, upper_bound = outlier_detection(X=df[col], h=1.5)
           n_outliers.append(len(outliers))
           prop_outliers.append(len(outliers)/n_rows)

        prop_not_outliers = [1 - x for x in prop_outliers]
        n_not_outliers = [n_rows - x for x in n_outliers]

        num_unique_values_quant_col = []
        for col in quant_col_names :
            unique_values = df[col].unique().to_numpy()
            unique_values = [x for x in unique_values if x is not None]
            num_unique_values_quant_col.append(len(unique_values))

        prop_nulls_quant_cols = prop_cols_nulls(df[quant_col_names])

        quant_summary = pd.DataFrame(columns=quant_col_names, 
                                 index=['n_unique', 'prop_nan', 'mean','std','min','Q10','Q25','median','Q75', 'Q90','max', 
                                       'kurtosis', 'skew', 'n_outliers', 'n_not_outliers', 'prop_outliers', 'prop_not_outliers'])

        quant_summary.loc['n_unique',:] = num_unique_values_quant_col
        quant_summary.loc['prop_nan',:] = prop_nulls_quant_cols
        quant_summary.loc['mean',:] = mean_quant_cols
        quant_summary.loc['std',:] = std_quant_cols
        quant_summary.loc['min',:] = min_quant_cols
        quant_summary.loc['Q10',:] = Q10_quant_cols
        quant_summary.loc['Q25',:] = Q25_quant_cols
        quant_summary.loc['median',:] = median_quant_cols 
        quant_summary.loc['Q75',:] = Q75_quant_cols
        quant_summary.loc['Q90',:] = Q90_quant_cols
        quant_summary.loc['max',:] = max_quant_cols
        quant_summary.loc['kurtosis',:] = kurtosis_quant_cols
        quant_summary.loc['skew',:] = skewness_quant_cols
        quant_summary.loc['n_outliers',:] = n_outliers
        quant_summary.loc['n_not_outliers',:] = n_not_outliers
        quant_summary.loc['prop_outliers',:] = prop_outliers
        quant_summary.loc['prop_not_outliers',:] = prop_not_outliers

    else :

        quant_summary = None


    if len(cat_col_names) > 0 :

        mode_cat_cols = [df[col].mode()[0] for col in cat_col_names]

        num_unique_values_cat_col = []
        for col in cat_col_names :
            unique_values = df[col].unique().to_numpy()
            unique_values = [x for x in unique_values if x is not None]
            num_unique_values_cat_col.append(len(unique_values))

        prop_nulls_cat_cols = prop_cols_nulls(df[cat_col_names])

        cat_summary = pd.DataFrame(columns=cat_col_names, 
                                   index=['n_unique', 'prop_nan', 'mode'])

        cat_summary.loc['n_unique',:] = num_unique_values_cat_col
        cat_summary.loc['prop_nan',:] = prop_nulls_cat_cols
        cat_summary.loc['mode',:] = mode_cat_cols

    else :

        cat_summary = None
  
    return quant_summary, cat_summary

######################################################################################################################

def cross_quant_cat_summary(df, quant_col, cat_col) :

    quant_cond_summary = df.group_by(cat_col).agg([
        (pl.col(quant_col).count() / len(df)).alias(f'prop_{quant_col}'),
        pl.col(quant_col).mean().alias(f'mean_{quant_col}'),
        pl.col(quant_col).std().alias(f'std_{quant_col}'),
        pl.col(quant_col).min().alias(f'min_{quant_col}'),
        pl.col(quant_col).quantile(0.10).alias(f'Q10_{quant_col}'),
        pl.col(quant_col).quantile(0.25).alias(f'Q25_{quant_col}'),
        pl.col(quant_col).median().alias(f'median_{quant_col}'),
        pl.col(quant_col).quantile(0.75).alias(f'Q75_{quant_col}'),
        pl.col(quant_col).quantile(0.90).alias(f'Q90_{quant_col}'),
        pl.col(quant_col).max().alias(f'max_price'),
        pl.col(quant_col).kurtosis().alias(f'kurtosis_{quant_col}'),
        pl.col(quant_col).skew().alias(f'skew_{quant_col}')
        ])
    
    quant_cond_summary = quant_cond_summary.filter(pl.col(cat_col).is_not_null())
    summary_columns = quant_cond_summary.columns
    quant_cond_summary_cat_col = quant_cond_summary[:,0].to_numpy()
    quant_cond_summary_rest = np.round(quant_cond_summary[:,1:].to_numpy(), 3)
    quant_cond_summary =  pl.from_numpy(np.column_stack((quant_cond_summary_cat_col, quant_cond_summary_rest)))
    quant_cond_summary.columns = summary_columns

    unique_values = df[cat_col].unique().to_numpy()
    unique_values = [x for x in unique_values if x is not None]

    df_cond = dict()
    df_cond[quant_col] = dict() 
    df_cond[quant_col][cat_col] = dict() 
    for cat in unique_values :
        df_cond[quant_col][cat_col][cat] = df.filter(pl.col(cat_col) == cat)[quant_col]

    prop_outliers_dict = dict()
    # prop_not_outliers_dict = dict()
    for cat in unique_values:
        trimmed_variable , outliers, lower_bound, upper_bound = outlier_detection(X=df_cond[quant_col][cat_col][cat], h=1.5)
        prop_outliers_dict[cat] = len(outliers)/len(df_cond[quant_col][cat_col][cat])
        # prop_not_outliers_dict[cat] = 1 - prop_outliers_dict[cat]

    prop_outliers = pl.Series(prop_outliers_dict.values())
    # prop_not_outliers = pl.Series(prop_not_outliers_dict.values())
    quant_cond_summary = quant_cond_summary.with_columns(prop_outliers.alias(f'prop_outliers_{quant_col}'))
    # quant_cond_summary = quant_cond_summary.with_columns(prop_not_outliers.alias('prop_not_outliers_buy_price'))

    prop_nulls_dict = dict()
    # prop_not_nulls_dict = dict()
    for cat in unique_values:
        prop_nulls_dict[cat] = prop_cols_nulls(df_cond[quant_col][cat_col][cat])
        # prop_not_nulls_dict[cat] = 1 - prop_nulls_dict[cat]

    prop_nulls = pl.Series(prop_nulls_dict.values())
    # prop_not_nulls = pl.Series(prop_not_nulls_dict.values())
    quant_cond_summary = quant_cond_summary.with_columns(prop_nulls.alias(f'prop_nan_{quant_col}'))
    # quant_cond_summary = quant_cond_summary.with_columns(prop_not_nulls.alias('prop_not_nan_buy_price'))
    if df[cat_col].dtype in [pl.Int32, pl.Int64, pl.Float32, pl.Float64]:
        quant_cond_summary = quant_cond_summary.sort(by=quant_cond_summary.columns[0])

    return quant_cond_summary

######################################################################################################################

def contingency_table_2D(df, cat1_name, cat2_name, conditional=False, axis=1) :

    # axis = 0: cat1 is the conditioning variable
    # axis = 1: cat2 is the conditioning variable

    if conditional == False :

       cat1_array = df[cat1_name].to_numpy().flatten()
       cat2_array = df[cat2_name].to_numpy().flatten()
       cat12_list = [(x,y) for (x,y) in zip(cat1_array, cat2_array)]

       count_dict = Counter(cat12_list)
       unique_values = count_dict.keys()
       counts = np.array([x for x in count_dict.values()])

       rel_counts = counts / len(df)
       name = f'({cat1_name}, {cat2_name}) : unique values'

       
       contigency_table_df = pl.DataFrame({name : unique_values, 
                                           'abs_freq' : counts, 
                                           'rel_freq' : np.round(rel_counts, 4), 
                                           'cum_abs_freq' : np.cumsum(counts), 
                                           'cum_rel_freq' : np.cumsum(rel_counts)})   

    elif conditional == True :

        cat12_cond = dict()
    
        if axis == 0 : 

            for cat in  np.unique(df[cat1_name]) :
                cat12_cond[cat] = df.filter(pl.col(cat1_name) == cat)[cat2_name]
            
            cat12_list = list()
            for cat in  np.unique(df[cat1_name]) :
                cat12_list = cat12_list + [(x,y) for (x,y) in product(cat12_cond[cat], [cat])]  

            count_dict = Counter(cat12_list)
            unique_values = count_dict.keys()
            counts = [x for x in count_dict.values()]
            rel_counts = list()
            for cat in np.unique(df[cat1_name]):
                rel_counts = rel_counts + [x / len(cat12_cond[cat]) for x,y in zip(count_dict.values(), count_dict.keys()) if y[1] == cat]
            name = f'({cat2_name} | {cat1_name}) : unique values'
        
        elif axis == 1 :

            for cat in  np.unique(df[cat2_name]) :
                cat12_cond[cat] = df.filter(pl.col(cat2_name) == cat)[cat1_name]
            
            cat12_list = list()
            for cat in  np.unique(df[cat2_name]) :
                cat12_list = cat12_list + [(x,y) for (x,y) in product(cat12_cond[cat], [cat])]  

            count_dict = Counter(cat12_list)
            unique_values = count_dict.keys()
            counts = np.array([x for x in count_dict.values()])
            rel_counts = list()
            for cat in np.unique(df[cat2_name]):
                rel_counts = rel_counts + [x / len(cat12_cond[cat]) for x,y in zip(count_dict.values(), count_dict.keys()) if y[1] == cat]         
            name = f'({cat1_name} | {cat2_name}) : unique values'


        contigency_table_df = pl.DataFrame({name : unique_values, 
                                           'abs_freq' : counts, 
                                           'rel_freq' : np.round(rel_counts, 4)})     
    
    return contigency_table_df

######################################################################################################################

def contingency_table_3D(df, cat1_name, cat2_name, cat3_name, conditional=False, axis=[1,2]) :

    if conditional == False :

       cat1_array = df[cat1_name].to_numpy().flatten()
       cat2_array = df[cat2_name].to_numpy().flatten()
       cat3_array = df[cat3_name].to_numpy().flatten()
       cat123_list = [(x,y,z) for (x,y,z) in zip(cat1_array, cat2_array, cat3_array)]

       count_dict = Counter(cat123_list)
       unique_values = count_dict.keys()
       counts = np.array([x for x in count_dict.values()])

       rel_counts = counts / len(df)
       name = f'({cat1_name}, {cat2_name}, {cat3_name}) : unique values'

       
       contigency_table_df = pl.DataFrame({name : unique_values, 
                                           'abs_freq' : counts, 
                                           'rel_freq' : np.round(rel_counts, 4), 
                                           'cum_abs_freq' : np.cumsum(counts), 
                                           'cum_rel_freq' : np.cumsum(rel_counts)})   

    elif conditional == True :

        cat123_cond = dict()
    
        if axis == [0,1] or axis == [1,0]: 

            cat123_cond = dict()

            for cat in  product(np.unique(df[cat1_name]), np.unique(df[cat2_name])) :
                cat123_cond[cat] = df.filter((pl.col(cat1_name) == cat[0]) & (pl.col(cat2_name) == cat[1]))[cat3_name]
        
            cat123_list = list()
            for cat in  product(np.unique(df[cat1_name]), np.unique(df[cat2_name])) :
                cat123_list = cat123_list + [(x,y,z) for x,(y,z) in product(cat123_cond[cat], [cat])]  

            count_dict = Counter(cat123_list)
            unique_values = count_dict.keys()
            counts = [x for x in count_dict.values()]
            rel_counts = list()

            for cat in product(np.unique(df[cat1_name]), np.unique(df[cat2_name])) :
                rel_counts = rel_counts + [x / len(cat123_cond[cat]) for x,y in zip(count_dict.values(), count_dict.keys()) if y[1:3] == cat]

            name = f'({cat3_name} | {cat1_name} , {cat2_name}) : unique values'
        
        if axis == [0,2] or axis == [2,0] : 

            cat123_cond = dict()

            for cat in  product(np.unique(df[cat1_name]), np.unique(df[cat3_name])) :
                cat123_cond[cat] = df.filter((pl.col(cat1_name) == cat[0]) & (pl.col(cat3_name) == cat[1]))[cat2_name]
        
            cat123_list = list()
            for cat in  product(np.unique(df[cat1_name]), np.unique(df[cat3_name])) :
                cat123_list = cat123_list + [(x,y,z) for x,(y,z) in product(cat123_cond[cat], [cat])]  

            count_dict = Counter(cat123_list)
            unique_values = count_dict.keys()
            counts = [x for x in count_dict.values()]
            rel_counts = list()

            for cat in product(np.unique(df[cat1_name]), np.unique(df[cat3_name])) :
                rel_counts = rel_counts + [x / len(cat123_cond[cat]) for x,y in zip(count_dict.values(), count_dict.keys()) if y[1:3] == cat]

            name = f'({cat2_name} | {cat1_name} , {cat3_name}) : unique values'

        if axis == [1,2] or axis == [2,1] : 

            cat123_cond = dict()

            for cat in  product(np.unique(df[cat2_name]), np.unique(df[cat3_name])) :
                cat123_cond[cat] = df.filter((pl.col(cat2_name) == cat[0]) & (pl.col(cat3_name) == cat[1]))[cat1_name]

            cat123_list = list()
            for cat in  product(np.unique(df[cat2_name]), np.unique(df[cat3_name])) :
                cat123_list = cat123_list + [(x,y,z) for x,(y,z) in product(cat123_cond[cat], [cat])]  

            count_dict = Counter(cat123_list)
            unique_values = count_dict.keys()
            counts = [x for x in count_dict.values()]
            rel_counts = list()

            for cat in product(np.unique(df[cat2_name]), np.unique(df[cat3_name])) :
                rel_counts = rel_counts + [x / len(cat123_cond[cat]) for x,y in zip(count_dict.values(), count_dict.keys()) if y[1:3] == cat]

            name = f'({cat1_name} | {cat2_name} , {cat3_name}) : unique values'


        contigency_table_df = pl.DataFrame({name : unique_values, 
                                           'abs_freq' : counts, 
                                           'rel_freq' : np.round(rel_counts, 4)})     
    
    return contigency_table_df

######################################################################################################################

def cov_matrix(df, auto_col=True, quant_col_names=None) :

    if auto_col == True :
        quant_col_names = columns_names(df, types=[pl.Float64, pl.Int64])

    p_quant = len(quant_col_names)
    cov_matrix_ = np.zeros((p_quant,p_quant))

    for i, col1 in enumerate(quant_col_names) :
        for j, col2 in enumerate(quant_col_names) :
            if j >= i:
               cov_matrix_[i,j] = np.round(madrid_houses_df.select(pl.cov(col1, col2)).to_numpy()[0][0], 2)

    cov_matrix_ = cov_matrix_ + np.triu(cov_matrix_, k=1).T

    cov_matrix_df = pd.DataFrame(cov_matrix_, columns=quant_col_names, index=quant_col_names)
    
    return cov_matrix_df

######################################################################################################################

def corr_matrix(df, auto_col=True, quant_col_names=None, response=None, predictors=None, method='pearson') :

    if response != None and predictors != None : 

        corr_list = []
        for col in predictors:
            corr_list.append(np.round(df.select(pl.corr(response, col, method=method)).to_numpy()[0][0], 2))

        corr_list_df = pd.DataFrame(corr_list, columns=[response], index=predictors) 

        return corr_list_df
    
    else :

        if auto_col == True :
            quant_col_names = columns_names(df, types=[pl.Float64, pl.Int64])

        p_quant = len(quant_col_names)
        corr_matrix_ = np.zeros((p_quant,p_quant))

        for i, col1 in enumerate(quant_col_names) :
            for j, col2 in enumerate(quant_col_names) :
                if j >= i:
                    corr_matrix_[i,j] = np.round(df.select(pl.corr(col1, col2, method=method)).to_numpy()[0][0], 2)

        corr_matrix_ = corr_matrix_ + np.triu(corr_matrix_, k=1).T

        corr_matrix_df = pd.DataFrame(corr_matrix_, columns=quant_col_names, index=quant_col_names) 
       
        return corr_matrix_df

######################################################################################################################

def high_corr(df, upper, lower, auto_col=True, quant_col_names=None, method='pearson'):

    if auto_col == True :
        quant_col_names = columns_names(df, types=[pl.Float64, pl.Int64])

    corr_dict = dict()

    for (col1,col2) in combinations(quant_col_names, 2) :
   
       corr = np.round(df.select(pl.corr(col1, col2, method=method)).to_numpy()[0][0], 2)
   
       if corr >= upper or corr <= lower :
      
          corr_dict[str((col1,col2))] = corr

    high_corr_df = pd.DataFrame(corr_dict, index=['corr'])

    return high_corr_df

######################################################################################################################

######################################################################################################################

######################################################################################################################
# PLOTS
######################################################################################################################

def get_ticks(min, max, n_ticks, n_round=2):
    step = (max - min) / (n_ticks - 1)
    ticks = np.arange(min, max, step)
    if ticks[-1] != max:
       ticks = np.append(ticks, max)
    ticks = np.round(ticks, n_round)
    return ticks

######################################################################################################################

def histogram(X, bins, color, figsize=(9,5), n_xticks=15, x_rotation=0, get_intervals=False, 
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid', n_round_xticks=2) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        
        X = X.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the histogram.
    p = sns.histplot(x=X, stat="proportion", bins=bins, color=color)

    # Setting the sticks for the histogram.
    min = np.floor(X.min())
    max = np.ceil(X.max())
    xticks = get_ticks(min, max, n_ticks=n_xticks, n_round=n_round_xticks)
    plt.xticks(xticks, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Histogram'+' - '+ X.name, fontsize=15)

    if get_intervals == True :

        interval = dict()
        for i, bar in enumerate(p.patches) :
            interval[i] = f'({bar.get_x()}, {bar.get_x() + bar.get_width()})'
        return interval
    
    if save == True :

        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

######################################################################################################################

def histogram_matrix(df, bins, n_cols, title, figsize=(15,15), auto_col=True, 
                     quant_col_names=[], remove_columns=[], add_columns=[], 
                     n_xticks=15, title_fontsize=15, subtitles_fontsize=11, save=False, 
                     file_name=None, random=False, n=None, fraction=None, seed=123, 
                     x_rotation=0, title_height=0.95, style='whitegrid', hspace=1, wspace=0.2,
                     n_round_xticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Selecting automatically the quantitative columns.
    if auto_col == True :
        quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])

        if len(remove_columns) > 0 :
            for r in remove_columns :
                quant_col_names.remove(r)

        if len(add_columns) > 0 : 
            for r in add_columns :
                quant_col_names.append(r)

    # Selecting automatically the quantitative columns.
    elif auto_col == False :
        quant_col_names = quant_col_names
   
    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(quant_col_names) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(quant_col_names))

    # Defining a ecdf-plot for each variable considered.
    for (i, col), color in zip(enumerate(quant_col_names), colors) :
      
        ax = axes[i]  # Get the current axis
        X = df.select(col).to_numpy().flatten()
        sns.histplot(data=X, stat="proportion", bins=bins, color=color, ax=ax)
        ax.set_title(col, fontsize=subtitles_fontsize)
        min = np.floor(df[col].min())
        max = np.ceil(df[col].max())
        xticks = get_ticks(min, max, n_ticks=n_xticks, n_round=n_round_xticks)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel('Proportion')

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(quant_col_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  

######################################################################################################################

def boxplot_old(X, color, figsize=(9,5), n_xticks=15, x_rotation=0, statistics=False, 
            random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
            style='whitegrid') :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :   
        X = X.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the histogram.
    p = sns.boxplot(x=X, color=color)

    # Setting the sticks for the histogram.
    min = np.floor(X.min())
    max = np.ceil(X.max())
    xticks_index = np.unique(np.round(np.linspace(min, max, n_xticks)))
    plt.xticks(xticks_index, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Boxplot'+' - '+ X.name, fontsize=15)

    if statistics == True :

        mean = X.mean()
        Q25 = X.quantile(0.25)
        Q50 = X.quantile(0.50)
        Q75 = X.quantile(0.75)
        max = X.max()
        min = X.min()
        IQR = Q75 - Q25
        oulier_lower_bound = Q25 - 1.5*IQR
        oulier_upper_bound = Q75 + 1.5*IQR

        plt.axvline(x=mean, c='skyblue', linestyle='dashed', label="mean")
        plt.axvline(x=Q50, c='green', linestyle='dashed', label="median")
        plt.axvline(x=Q25, c='orange', linestyle='dashed', label="Q25")
        plt.axvline(x=Q75, c='gold', linestyle='dashed', label="Q75")
        plt.axvline(x=min, c='pink', linestyle='dashed', label="min")
        plt.axvline(x=max, c='purple', linestyle='dashed', label="max")
        plt.axvline(x=oulier_lower_bound, c='red', linestyle='dashed', label="Oulier_lower_bound")
        plt.axvline(x=oulier_upper_bound, c='blue', linestyle='dashed', label="Oulier_lower_bound")

        labels = ["mean", "median", "Q25", "Q75", "min", "max", "Q25 - 1.5*IQR", "Q75 + 1.5*IQR"]
        handles, _ = p.get_legend_handles_labels()

        plt.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.2, 1))

    if save == True :

        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()
######################################################################################################################
    
def boxplot(X, color, figsize=(9,5), n_xticks=15, x_rotation=0, statistics=None, 
            random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
            style='whitegrid', lines_width=0.55, bbox_to_anchor=(0.5,-0.5), legend_size=10,
              color_stats=None) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :   
        X = X.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the histogram.
    p = sns.boxplot(x=X, color=color)

    # Setting the sticks for the histogram.
    min = np.floor(X.min())
    max = np.ceil(X.max())
    xticks_index = np.unique(np.round(np.linspace(min, max, n_xticks)))
    plt.xticks(xticks_index, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Boxplot'+' - '+ X.name, fontsize=15)

    if statistics is not None :

        n_statistics = len(statistics)
        if color_stats is None:
            color_stats = sns.color_palette("tab10", n_statistics)
        color_dict = {stat : color for color, stat in zip(color_stats, statistics)}

        if 'median' in statistics :
            median = X.median()
            plt.vlines(x=median, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['median'], linestyles='dashed', label='median', zorder=4)

        if 'mean' in statistics :
            mean = X.mean()
            plt.vlines(x=mean, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['mean'], linestyles='dashed', label='mean', zorder=4)

        if 'Q25' in statistics :
            Q25 = X.quantile(0.25)
            plt.vlines(x=Q25, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['Q25'], linestyles='dashed', label=f'Q25', zorder=4)

        if 'Q75' in statistics :
            Q75 = X.quantile(0.75)
            plt.vlines(x=Q75, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['Q75'], linestyles='dashed', label=f'Q75', zorder=4)

        handles, _ = p.get_legend_handles_labels()
        plt.legend(handles=handles, labels=statistics,  loc='lower center', bbox_to_anchor=bbox_to_anchor, 
                      ncol=n_statistics, fontsize=legend_size)

    if save == True :

        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

######################################################################################################################


def boxplot_matrix_old(df, n_cols, tittle, figsize=(15,15), auto_col=True, 
                     quant_col_names=[], remove_columns=[], add_columns=[], 
                     n_xticks=15, fontsize=15, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, title_height=0.95,
                     style='whitegrid', hspace=1, wspace=0.2, subtitles_fontsize=12) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Selecting automatically the quantitative columns.
    if auto_col == True :
        quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])

        if len(remove_columns) > 0 :
            for r in remove_columns :
                quant_col_names.remove(r)

        if len(add_columns) > 0 : 
            for r in add_columns :
                quant_col_names.append(r)

    # Selecting automatically the quantitative columns.
    elif auto_col == False :
        quant_col_names = quant_col_names
   
    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(quant_col_names) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(quant_col_names))

    # Defining a ecdf-plot for each variable considered.
    for (i, col), color in zip(enumerate(quant_col_names), colors) :
      
        ax = axes[i]  # Get the current axis
        X = df.select(col).to_numpy().flatten()
        sns.boxplot(x=X, color=color, ax=ax)
        ax.set_title(col, fontsize=subtitles_fontsize)
        min = np.floor(df[col].min())
        max = np.ceil(df[col].max())

        xticks_index = np.unique(np.round(np.linspace(min, max, n_xticks)))
        plt.xticks(xticks_index) 
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel('')

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(quant_col_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(tittle, fontsize=fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()

######################################################################################################################
    
def boxplot_matrix(df, n_cols, title, figsize=(15,15), auto_col=True, 
                     quant_col_names=[], remove_columns=[], add_columns=[], 
                     n_xticks=10, title_fontsize=15, subtitles_fontsize=12, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, title_height=0.95,
                     style='whitegrid', hspace=1, wspace=0.2, statistics=None, lines_width=0.55, 
                     bbox_to_anchor=(0.5,-0.5), legend_size=10, color_stats=None, n_round_xticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Selecting automatically the quantitative columns.
    if auto_col == True :
        quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])

        if len(remove_columns) > 0 :
            for r in remove_columns :
                quant_col_names.remove(r)

        if len(add_columns) > 0 : 
            for r in add_columns :
                quant_col_names.append(r)

    # Selecting automatically the quantitative columns.
    elif auto_col == False :
        quant_col_names = quant_col_names
   
    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(quant_col_names) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(quant_col_names))

    # Defining a ecdf-plot for each variable considered.
    for (i, col), color in zip(enumerate(quant_col_names), colors) :
      
        ax = axes[i]  # Get the current axis
        X = df.select(col).to_numpy().flatten()
        sns.boxplot(x=X, color=color, ax=ax)
        ax.set_title(col, fontsize=subtitles_fontsize)
        min = df[col].min()
        max = df[col].max()
        xticks = get_ticks(min, max, n_ticks=n_xticks, n_round=n_round_xticks)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel('')

        if statistics is not None :

           n_statistics = len(statistics)
           if color_stats is None:
               color_stats = sns.color_palette("tab10", n_statistics)
           color_dict = {stat : color for color, stat in zip(color_stats, statistics)}

           if 'median' in statistics :
               median = np.median(X)
               ax.vlines(x=median, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['median'], 
                          linestyles='dashed', label='median', zorder=4)

           if 'mean' in statistics :
               mean = np.mean(X)
               ax.vlines(x=mean, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['mean'], 
                          linestyles='dashed', label='mean', zorder=4)

           if 'Q25' in statistics :
               Q25 = np.quantile(X, 0.25)
               ax.vlines(x=Q25, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['Q25'], 
                          linestyles='dashed', label=f'Q25', zorder=4)

           if 'Q75' in statistics :
               Q75 = np.quantile(X, 0.75)
               ax.vlines(x=Q75, ymin=-0.1 - lines_width/2, ymax=0.1 + lines_width/2, colors=color_dict['Q75'], 
                          linestyles='dashed', label=f'Q75', zorder=4)

           handles, _ = ax.get_legend_handles_labels() 
           fig.legend(handles, statistics, loc='lower center', 
                       bbox_to_anchor=bbox_to_anchor, ncol=len(statistics), fontsize=legend_size)

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(quant_col_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()

######################################################################################################################

def ecdfplot(X, color, figsize=(9,5), n_xticks=15, n_yticks=10, x_rotation=0, y_rotation=0, complementary=False, 
            random=False, n=None, fraction=None, seed=123, save=False, file_name=None, n_round_xticks=2) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    if random == True :
        
        X = X.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the histogram.
    p = sns.ecdfplot(x=X, color=color, complementary=complementary)

    # Setting the sticks for the histogram.
    min = np.floor(X.min())
    max = np.ceil(X.max())
    xticks = get_ticks(min, max, n_ticks=n_xticks, n_round=n_round_xticks)
    yticks = np.unique(np.round(np.linspace(0, 1, n_yticks), 2))
    plt.xticks(xticks, rotation=x_rotation) 
    plt.yticks(yticks, rotation=y_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'ECDFplot'+' - '+ X.name, fontsize=15)

    if save == True :

        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

######################################################################################################################

def ecdf_matrix(df, n_cols, title, complementary=False, figsize=(15,15), auto_col=True, 
                     quant_col_names=[], remove_columns=[], add_columns=[], 
                     n_xticks=15, title_fontsize=15, subtitles_fontsize=11, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, title_height=0.95,
                     style='whitegrid', hspace=1, wspace=0.2, n_round_xticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Selecting automatically the quantitative columns.
    if auto_col == True :
        quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])

        if len(remove_columns) > 0 :
            for r in remove_columns :
                quant_col_names.remove(r)

        if len(add_columns) > 0 : 
            for r in add_columns :
                quant_col_names.append(r)

    # Selecting automatically the quantitative columns.
    elif auto_col == False :
        quant_col_names = quant_col_names
   
    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(quant_col_names) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(quant_col_names))

    # Defining a ecdf-plot for each variable considered.
    for (i, col), color in zip(enumerate(quant_col_names), colors) :
      
        ax = axes[i]  # Get the current axis
        X = df.select(col).to_numpy().flatten()
        sns.ecdfplot(x=X, color=color, complementary=complementary, ax=ax)
        ax.set_title(col, fontsize=subtitles_fontsize)
        min = np.floor(df[col].min())
        max = np.ceil(df[col].max())
        xticks = get_ticks(min, max, n_ticks=n_xticks, n_round=n_round_xticks)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel('')

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(quant_col_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()

######################################################################################################################

def barplot(X, color, categories_order=None, figsize=(9,5), xticks_rotation=0, random=False, 
            n=None, fraction=None, seed=123, fontsize=10, y_up_limit=1, 
            y_low_limit=0, xticks_fontsize=11, yticks_fontsize=11, xlabel_size=11, 
            ylabel_size=11, ylabel='Relative Frequency', xlabel='',
            title_size=14, title_weight='bold') :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    color: name of the color to be use for the histogram bars.
    categories_order: a list with X categories order as they will appear in the plot.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A bar-plot of X variable, with the parameters specified.
    """

    # To use this plot we need a Pandas series, 
    # because X.value_counts(normalize=True).reindex(categories_order) 
    # doesn't work well with Polars.

    if random == True :
        X = X.sample(fraction=fraction, n=n, seed=seed)

    if isinstance(X, pl.Series):
        X = X.to_pandas()

    # Setting the figure size.
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the barplot.
    value_counts = X.value_counts(normalize=True).reindex(categories_order)
    ax = value_counts.plot(kind='bar', color=color)

    ax.set_ylabel(ylabel, size=ylabel_size)
    ax.set_xlabel(xlabel, size=xlabel_size)
    plt.xticks(fontsize=xticks_fontsize, rotation=xticks_rotation)
    plt.yticks(fontsize=yticks_fontsize)

    # Setting the title of the plot.
    plt.title(label = 'Bar-plot' + '  ' + X.name, fontsize=title_size, weight=title_weight)
    ax.set_ylim(y_low_limit, y_up_limit)  

    # Add text annotations to each bar
    for i, v in enumerate(value_counts):
        plt.text(i, v, f"{v:.2f}", color='black', ha='center', va='bottom', fontsize=fontsize, fontweight='bold')

    plt.show()


######################################################################################################################

def barplot_matrix(df, n_cols, title, figsize=(15,15), auto_col=True, 
                     cat_col_list=[], remove_columns=[], add_columns=[], 
                     title_fontsize=15, subtitles_fontsize=12, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, title_height=0.95,
                     style='whitegrid', hspace=1, wspace=0.5, n_yticks=4, n_round_yticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Selecting automatically the categorical columns.
    if auto_col == True :
        cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])

        if len(remove_columns) > 0 :
            for r in remove_columns :
                cat_col_names.remove(r)

        if len(add_columns) > 0 : 
            for r in add_columns :
                cat_col_names.append(r)

    # Selecting automatically the quantitative columns.
    elif auto_col == False :
        cat_col_names = cat_col_list
   
    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cat_col_names) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(cat_col_names))

    # Defining a ecdf-plot for each variable considered.
    for (i, col), color in zip(enumerate(cat_col_names), colors) :
      
        ax = axes[i]  # Get the current axis
        X = df[col].to_pandas()
        value_counts = X.value_counts(normalize=True).reindex()
        value_counts.plot(kind='bar', color = color, ax=ax)
        ax.set_title(col, fontsize=subtitles_fontsize)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col)
        ax.set_ylabel('Proportion')
        yticks = get_ticks(0, np.max(value_counts)+0.05, n_ticks=n_yticks, n_round=n_round_yticks)
        ax.set_yticks(yticks)

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cat_col_names), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()

######################################################################################################################

def scatter(X, Y, color, figsize=(9,5), n_xticks=10, n_yticks=10, x_rotation=0,   
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid') :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :     
        X = X.sample(fraction=fraction, n=n, seed=seed)
        Y = Y.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the histogram.
    p = sns.scatterplot(x=X, y=Y, color=color)

    # Setting the sticks for x-axis.
    X_min = np.floor(X.min())
    X_max = np.ceil(X.max())
    xticks_index = np.unique(np.round(np.linspace(X_min, X_max, n_xticks)))
    plt.xticks(xticks_index, rotation=x_rotation) 
    
    # Setting the sticks for y-axis.
    Y_min = np.floor(Y.min())
    Y_max = np.ceil(Y.max())
    yticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_yticks)))
    plt.yticks(yticks_index, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Scatter'+' - '+ X.name + '-' + Y.name, fontsize=15)
    
    if save == True :
        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()


######################################################################################################################

def scatter_matrix(df, n_cols, title, figsize=(15,15), auto_col=True, 
                     response=None, predictors=None,
                     quant_col_names=[], remove_columns=[], add_columns=[], 
                     n_xticks=10, n_yticks=10, title_fontsize=15, subtitles_fontsize=12, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, y_rotation=0,
                     title_height=0.95, style='whitegrid', hspace=1, wspace=0.2, n_round_xticks=2, n_round_yticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)


    if response != None and predictors != None : 

        cols_combis = list(product(predictors, response))

    else :

        # Selecting automatically the quantitative columns.
        if auto_col == True :
            quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])

            if len(remove_columns) > 0 :
                for r in remove_columns :
                    quant_col_names.remove(r)

            if len(add_columns) > 0 : 
                for r in add_columns :
                    quant_col_names.append(r)
   
        cols_combis = list(combinations(quant_col_names, 2))


    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cols_combis) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(cols_combis))

    # Defining a ecdf-plot for each variable considered.
    for (i, (col1, col2)), color in zip(enumerate(cols_combis), colors) :
        
        ax = axes[i]  # Get the current axis
        X = df.select(col1).to_numpy().flatten()
        Y = df.select(col2).to_numpy().flatten()
        sns.scatterplot(x=X, y=Y, color=color, ax=ax)
        ax.set_title(col2 + ' vs ' + col1, fontsize=subtitles_fontsize)
        X_min = np.floor(df[col1].min())
        X_max = np.ceil(df[col1].max())
        Y_min = np.floor(df[col2].min())
        Y_max = np.ceil(df[col2].max())  
        xticks = get_ticks(X_min, X_max, n_ticks=n_xticks, n_round=n_round_xticks)
        yticks = get_ticks(Y_min, Y_max, n_ticks=n_yticks, n_round=n_round_yticks)
        ax.set_yticks(yticks)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.tick_params(axis='y', rotation=y_rotation)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cols_combis), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  

######################################################################################################################

def stripplot(df, X_name, Y_name, color, jitter=0.15, figsize=(9,5), n_yticks=10, x_rotation=0,   
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid', size=3, statistics=None, lines_width=0.55, bbox_to_anchor=(0.5,-0.5), legend_size=10,
              color_stats=None) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :     
        df = df.sample(fraction=fraction, n=n, seed=seed)

    X = df[X_name]
    Y = df[Y_name]

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the plot.
    p = sns.stripplot(x=X, y=Y, color=color, jitter=jitter, size=size)

    # Setting the sticks for x-axis.
    plt.xticks(rotation=x_rotation) 
    
    # Setting the sticks for y-axis.
    Y_min = np.floor(Y.min())
    Y_max = np.ceil(Y.max())
    yticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_yticks)))
    plt.yticks(yticks_index, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Stripplot'+' - '+ Y.name + ' vs ' + X.name, fontsize=15)

    if statistics is not None :

        n_statistics = len(statistics)
        median, median[Y_name], median[Y_name][X_name] = {}, {}, {}
        mean, mean[Y_name], mean[Y_name][X_name] = {}, {}, {}
        if color_stats is None:
            color_stats = sns.color_palette("tab10", n_statistics*len(df[X_name].unique()))
        color_dict = {stat : color for color, stat in zip(color_stats, statistics)}
        #stats_labels = []

        # Get the x-axis category positions
        category_labels = p.get_xticklabels()  # This gives the order seaborn is using for categories
        category_positions = p.get_xticks()
        pos_dict = {label.get_text(): pos for label, pos in zip(category_labels, category_positions)}
    
        for i, cat in enumerate(df[X_name].unique()) :
           cat_pos = pos_dict[str(cat)]   
           width_of_category = lines_width  # Adjust as necessary to match the width of your categories

           if 'median' in statistics :
               median[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].median()
               plt.hlines(y=median[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['median'], linestyles='dashed', label=f'median_{cat}', zorder=4)
               #stats_labels.append(f'median_{Y_name}_{X_name}={cat}')

           if 'mean' in statistics :
               mean[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].mean()
               plt.hlines(y=mean[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['mean'], linestyles='dashed', label=f'mean_{cat}', zorder=4)
               #stats_labels.append(f'mean_{Y_name}_{X_name}={cat}') 

        handles, _ = p.get_legend_handles_labels()
        plt.legend(handles=handles, labels=statistics,  loc='lower center', bbox_to_anchor=bbox_to_anchor, 
                      ncol=n_statistics, fontsize=legend_size)

    if save == True :
        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()


######################################################################################################################


def stripplot_matrix(df, n_cols, title, figsize=(15,15), auto_col=True, 
                     response=None, predictors=None, quant_col_names=[], cat_col_names=[], remove_quant_col=[], add_quant_col=[], 
                     remove_cat_col=[], add_cat_col=[], jitter=0.10, size=3.5, n_yticks=10, 
                     title_fontsize=15, subtitles_fontsize=12, save=False, file_name=None,
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, y_rotation=0,
                     title_height=0.95, style='whitegrid', hspace=1, wspace=0.2, statistics=None, lines_width=0.5, 
                     bbox_to_anchor=(0.5,-1), legend_size=9, color_stats=None, n_round_yticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    if response != None and predictors != None :

        cols_combis = list(product(predictors, response))

    else :

        # Selecting automatically the quantitative columns.
        if auto_col == True :
            quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])
            cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])

            if len(remove_quant_col) > 0 :
                for r in remove_quant_col :
                   quant_col_names.remove(r)

            if len(remove_cat_col) > 0 :
                for r in remove_cat_col :
                   cat_col_names.remove(r)
                
            if len(add_quant_col) > 0 : 
                for r in add_quant_col :
                    quant_col_names.append(r)

            if len(add_cat_col) > 0 : 
                for r in add_cat_col :
                    cat_col_names.append(r)

        cols_combis = list(product(cat_col_names, quant_col_names))


    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cols_combis) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(cols_combis))

    # Defining a ecdf-plot for each variable considered.
    for (i, (col1, col2)), color in zip(enumerate(cols_combis), colors) :
        
        ax = axes[i]  # Get the current axis
        X = df.select(col1).to_numpy().flatten()
        Y = df.select(col2).to_numpy().flatten()
        sns.stripplot(x=X, y=Y, color=color, jitter=jitter, size=size, ax=ax)
        ax.set_title(col2 + ' vs ' + col1, fontsize=subtitles_fontsize)
        Y_min = np.floor(df[col2].min())
        Y_max = np.ceil(df[col2].max()) 
        yticks = get_ticks(Y_min, Y_max, n_ticks=n_yticks, n_round=n_round_yticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.tick_params(axis='y', rotation=y_rotation)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

        if statistics is not None :

            X_name = col1 ; Y_name = col2
            n_statistics = len(statistics)
            median, median[Y_name], median[Y_name][X_name] = {}, {}, {}
            mean, mean[Y_name], mean[Y_name][X_name] = {}, {}, {}
            if color_stats is None:
                color_stats = sns.color_palette("tab10", n_statistics*len(df[X_name].unique()))
            color_dict = {stat : color for color, stat in zip(color_stats, statistics)}
            #stats_labels = []

            # Get the x-axis category positions
            category_labels = ax.get_xticklabels()  # This gives the order seaborn is using for categories
            category_positions = ax.get_xticks()
            pos_dict = {label.get_text(): pos for label, pos in zip(category_labels, category_positions)}
    
            for cat in df[X_name].unique() :
                
                cat_pos = pos_dict[str(cat)]   
                width_of_category = lines_width  # Adjust as necessary to match the width of your categories

                if 'median' in statistics :
                    median[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].median()
                    ax.hlines(y=median[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['median'], linestyles='dashed', label=f'median_{cat}', zorder=4)
                    ax.get_legend().remove() 

                if 'mean' in statistics :
                   mean[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].mean()
                   ax.hlines(y=mean[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['mean'], linestyles='dashed', label=f'mean_{cat}', zorder=4)
                   #ax.get_legend().remove() 

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cols_combis), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    handles, labels = ax.get_legend_handles_labels() 
    fig.legend(handles, statistics, loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=len(statistics), fontsize=legend_size)

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  

######################################################################################################################

def boxplot_2D(df, X_name, Y_name, color, figsize=(9,5), n_yticks=10, x_rotation=0,   
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid', statistics=None, lines_width=0.55, bbox_to_anchor=(0.5,-0.5), legend_size=10,
              color_stats=None) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :     
        df = df.sample(fraction=fraction, n=n, seed=seed)

    X = df[X_name]
    Y = df[Y_name]

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)

    # Computing the plot.
    p = sns.boxplot(x=X, y=Y, color=color)

    # Setting the sticks for x-axis.
    plt.xticks(rotation=x_rotation) 
    
    # Setting the sticks for y-axis.
    Y_min = np.floor(Y.min())
    Y_max = np.ceil(Y.max())
    yticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_yticks)))
    plt.yticks(yticks_index, rotation=x_rotation) 

    # Setting the title of the plot.
    plt.title(label = 'Boxplot'+' - '+ Y.name + ' vs ' + X.name, fontsize=15)

    if statistics is not None :

        n_statistics = len(statistics)
        median, median[Y_name], median[Y_name][X_name] = {}, {}, {}
        mean, mean[Y_name], mean[Y_name][X_name] = {}, {}, {}
        if color_stats is None:
            color_stats = sns.color_palette("tab10", n_statistics*len(df[X_name].unique()))
        color_dict = {stat : color for color, stat in zip(color_stats, statistics)}
        #stats_labels = []

        # Get the x-axis category positions
        category_labels = p.get_xticklabels()  # This gives the order seaborn is using for categories
        category_positions = p.get_xticks()
        pos_dict = {label.get_text(): pos for label, pos in zip(category_labels, category_positions)}
    
        for cat in df[X_name].unique() :
           cat_pos = pos_dict[str(cat)]   
           width_of_category = lines_width  # Adjust as necessary to match the width of your categories

           if 'median' in statistics :
               median[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].median()
               plt.hlines(y=median[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['median'], linestyles='dashed', label=f'median_{cat}', zorder=4)
               #stats_labels.append(f'median_{Y_name}_{X_name}={cat}')

           if 'mean' in statistics :
               mean[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].mean()
               plt.hlines(y=mean[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['mean'], linestyles='dashed', label=f'mean_{cat}', zorder=4)
               #stats_labels.append(f'mean_{Y_name}_{X_name}={cat}') 

        handles, _ = p.get_legend_handles_labels()
        plt.legend(handles=handles, labels=statistics,  loc='lower center', bbox_to_anchor=bbox_to_anchor, 
                      ncol=n_statistics, fontsize=legend_size)

    if save == True :
        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

######################################################################################################################

def boxplot_2D_matrix(df, n_cols, tittle, figsize=(15,15), auto_col=False, 
                     response=None, predictors=None,
                     quant_col_names=[], cat_col_names=[], remove_quant_col=[], add_quant_col=[], 
                     remove_cat_col=[], add_cat_col=[], n_yticks=10, 
                     title_fontsize=15, subtitles_fontsize=12, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, y_rotation=0,
                     title_height=0.95, style='whitegrid', hspace=1, wspace=0.2, statistics=None, lines_width=0.5, 
                     bbox_to_anchor=(0.5,-1), legend_size=9, color_stats=None, showfliers = True, n_round_yticks=2) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    if response != None and predictors != None : 

        cols_combis = list(product(predictors, response))

    else :

        # Selecting automatically the quantitative columns.
        if auto_col == True :
            quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])
            cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])

            if len(remove_quant_col) > 0 :
                for r in remove_quant_col :
                    quant_col_names.remove(r)

            if len(remove_cat_col) > 0 :
                for r in remove_cat_col :
                   cat_col_names.remove(r)
                
            if len(add_quant_col) > 0 : 
                for r in add_quant_col :
                    quant_col_names.append(r)

            if len(add_cat_col) > 0 : 
                for r in add_cat_col :
                    cat_col_names.append(r)

        cols_combis = list(product(cat_col_names, quant_col_names))


    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cols_combis) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab10", len(cols_combis))

    # Defining a ecdf-plot for each variable considered.
    for (i, (col1, col2)), color in zip(enumerate(cols_combis), colors) :
        
        ax = axes[i]  # Get the current axis
        X = df.select(col1).to_numpy().flatten()
        Y = df.select(col2).to_numpy().flatten()
        sns.boxplot(x=X, y=Y, color=color, showfliers=showfliers, ax=ax)
        ax.set_title(col2 + ' vs ' + col1, fontsize=subtitles_fontsize)
        if showfliers == True :
            Y_min = np.floor(df[col2].min())
            Y_max = np.ceil(df[col2].max())
            yticks = get_ticks(Y_min, Y_max, n_ticks=n_yticks, n_round=n_round_yticks)        
            ax.set_yticks(yticks)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.tick_params(axis='y', rotation=y_rotation)
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)

        if statistics is not None :

            X_name = col1 ; Y_name = col2
            n_statistics = len(statistics)
            median, median[Y_name], median[Y_name][X_name] = {}, {}, {}
            mean, mean[Y_name], mean[Y_name][X_name] = {}, {}, {}
            if color_stats is None:
                color_stats = sns.color_palette("tab10", n_statistics*len(df[X_name].unique()))
            color_dict = {stat : color for color, stat in zip(color_stats, statistics)}
            #stats_labels = []

            # Get the x-axis category positions
            category_labels = ax.get_xticklabels()  # This gives the order seaborn is using for categories
            category_positions = ax.get_xticks()
            pos_dict = {label.get_text(): pos for label, pos in zip(category_labels, category_positions)}
    
            for cat in df[X_name].unique() :
                
                cat_pos = pos_dict[str(cat)]   
                width_of_category = lines_width  # Adjust as necessary to match the width of your categories

                if 'median' in statistics :
                    median[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].median()
                    ax.hlines(y=median[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['median'], linestyles='dashed', label=f'median_{cat}', zorder=4)

                if 'mean' in statistics :
                   mean[Y_name][X_name][cat] = df.filter(pl.col(X_name) == cat)[Y_name].mean()
                   ax.hlines(y=mean[Y_name][X_name][cat], xmin=cat_pos-width_of_category/2, xmax=cat_pos+width_of_category/2, 
                          colors=color_dict['mean'], linestyles='dashed', label=f'mean_{cat}', zorder=4)

            handles, labels = ax.get_legend_handles_labels() 
            fig.legend(handles, statistics, loc='lower center', bbox_to_anchor=bbox_to_anchor, ncol=len(statistics), fontsize=legend_size)

    # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cols_combis), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(tittle, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  

#########################################################################################################
    
def histogram_2D(df, quant_column, cat_column, bins=10, figsize=(9,5), n_yticks=5, n_xticks=10, x_rotation=0,   
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid', bbox_to_anchor=(1,1), legend_size=10, transparency=0.8) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :     
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)
    patches = []
    cat_unique_values = df[cat_column].unique()
    colors = sns.color_palette("tab10", len(cat_unique_values))
    
    # Computing the plot.
    for value, color in zip(cat_unique_values, colors):
        Y_cond = df.filter(pl.col(cat_column)==value).select(quant_column).to_numpy().flatten()
        p = sns.histplot(x=Y_cond, bins=bins, color=color, alpha=transparency, stat='proportion')
        patch = mpatches.Patch(color=color, label=value)
        patches.append(patch)
    p.set_title(quant_column + ' vs ' + cat_column, fontsize=13)
    Y_min = np.floor(df[quant_column].min())
    Y_max = np.ceil(df[quant_column].max())        
    xticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_xticks)))
    p.set_xticks(xticks_index)
    p.set_yticks(np.linspace(0, 1, n_yticks))
    p.tick_params(axis='x', rotation=x_rotation)
    p.set_xlabel(quant_column)
    p.set_ylabel('Proportion')
    p.legend(handles=patches, title=cat_column, loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_size)
    
    # Setting the title of the plot.
    plt.title(label = 'Histogram '+' - '+ quant_column + ' vs ' + cat_column, fontsize=15)

    if save == True :
        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

######################################################################################################################
    
def histogram_2D_matrix(df, bins, n_cols, tittle, figsize=(15,15), auto_col=True, 
                     response=None, predictors=None,
                     quant_col_names=[], cat_col_names=[], remove_quant_col=[], add_quant_col=[], 
                     remove_cat_col=[], add_cat_col=[], n_yticks=5, n_xticks=10,
                     title_fontsize=15, subtitles_fontsize=13, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, 
                     title_height=0.95, style='whitegrid', hspace=1, wspace=0.2,   
                     bbox_to_anchor=(1,1), legend_size=10, transparency=0.8) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    if response != None and predictors != None : 

        cols_combis = list(product(predictors, response))

    else :

        # Selecting automatically the quantitative columns.
        if auto_col == True :
            quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])
            cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])

            if len(remove_quant_col) > 0 :
                for r in remove_quant_col :
                    quant_col_names.remove(r)

            if len(remove_cat_col) > 0 :
                for r in remove_cat_col :
                   cat_col_names.remove(r)
                
            if len(add_quant_col) > 0 : 
                for r in add_quant_col :
                    quant_col_names.append(r)

            if len(add_cat_col) > 0 : 
                for r in add_cat_col :
                    cat_col_names.append(r)

        cols_combis = list(product(cat_col_names, quant_col_names))


    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cols_combis) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab20", 20)

    # Defining a ecdf-plot for each variable considered.
    for i, (col1, col2) in enumerate(cols_combis) :

        patches = []
        ax = axes[i]  # Get the current axis
        #X = df.select(col1).to_numpy().flatten()
        sorted_unique_values = np.sort(df[col1].unique())
        for value, color in zip(sorted_unique_values, colors):
            Y_cond = df.filter(pl.col(col1)==value).select(col2).to_numpy().flatten()
            sns.histplot(x=Y_cond, stat="proportion", bins=bins, color=color, ax=ax, alpha=transparency)
            patch = mpatches.Patch(color=color, label=value)
            patches.append(patch)
        ax.set_title(col2 + ' vs ' + col1, fontsize=subtitles_fontsize)
        Y_min = np.floor(df[col2].min())
        Y_max = np.ceil(df[col2].max())        
        xticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_xticks)))
        ax.set_yticks(np.linspace(0, 1, n_yticks))
        ax.set_xticks(xticks_index)
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col2)
        ax.set_ylabel('Proportion')
        ax.legend(handles=patches, title=col1, loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_size)

     # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cols_combis), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(tittle, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  

#########################################################################################################
    
def ecdf_2D(df, quant_column, cat_column, complementary=False, figsize=(9,5), n_yticks=5, n_xticks=10, x_rotation=0,   
              random=False, n=None, fraction=None, seed=123, save=False, file_name=None,
              style='whitegrid', bbox_to_anchor=(1,1), legend_size=10,
              transparency=0.8) :

    """
    Parameters (inputs)
    ----------
    X: a pandas series or a numpy array (the variable).
    bins: number of intervals used to create the histogram (number of bars).
    color: name of the color to be use for the histogram bars.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    rotation: a integer positive number. Indicates the rotation degree of the sticks from the axis.
    get_intervals:If True, the intervals used to create the histogram will be return. If False, not.
    sep: a parameter used for creating the sticks of the x-axis. We recommend using the default value.
   
    Returns (outputs)
    ----------
    A histogram of X variable, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :     
        df = df.sample(fraction=fraction, n=n, seed=seed)

    # Setting the figure size
    fig, axs = plt.subplots(figsize=figsize)
    patches = []
    cat_unique_values = df[cat_column].unique()
    colors = sns.color_palette("tab10", len(cat_unique_values))
    
    # Computing the plot.
    for value, color in zip(cat_unique_values, colors):
        Y_cond = df.filter(pl.col(cat_column)==value).select(quant_column).to_numpy().flatten()
        p = sns.ecdfplot(x=Y_cond, color=color, alpha=transparency, complementary=complementary)
        patch = mpatches.Patch(color=color, label=value)
        patches.append(patch)
    p.set_title(quant_column + ' vs ' + cat_column, fontsize=13)
    Y_min = np.floor(df[quant_column].min())
    Y_max = np.ceil(df[quant_column].max())        
    xticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_xticks)))
    p.set_xticks(xticks_index)
    p.set_yticks(np.linspace(0, 1, n_yticks))
    p.tick_params(axis='x', rotation=x_rotation)
    p.set_xlabel(quant_column)
    p.set_ylabel('Proportion')
    p.legend(handles=patches, title=cat_column, loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_size)
    
    # Setting the title of the plot.
    plt.title(label = 'Ecdfplot'+' - '+ quant_column + ' vs ' + cat_column, fontsize=15)

    if save == True :
        fig.savefig(file_name + '.jpg', format='jpg', dpi=600, bbox_inches="tight")
    
    plt.show()

#########################################################################################################
    
def ecdf_2D_matrix(df, n_cols, title, complementary=False, figsize=(15,15), auto_col=True, 
                     response=None, predictors=None,
                     quant_col_names=[], cat_col_names=[], remove_quant_col=[], add_quant_col=[], 
                     remove_cat_col=[], add_cat_col=[], n_yticks=5, n_xticks=10,
                     title_fontsize=15, subtitle_fontsize=12, save=False, file_name=None, 
                     random=False, n=None, fraction=None, seed=123, x_rotation=0, 
                     title_height=0.95, style='whitegrid', hspace=1, wspace=0.2,   
                     bbox_to_anchor=(1,1), legend_size=10, transparency=0.8) :
 
    """
    Parameters (inputs)
    ----------
    df: a polars data-frame (the data-matrix).
    bins: number of intervals used to create the histogram (number of bars).
    tittle: the tittle of the histogram.
    figsize: dimensions of the plot. Must be a pair of numbers (a,b), where a indicates the plot width, and b the length.
    auto_col: if True, the quantitative columns are selected automatically. If False, the function uses the columns of col_list.
    auto_dim: if True, the matrix-plot dimension is defined automatically. If False, the function uses (n,m) as dimension.
    n, m: number of rows (n) and columns (m) of the matrix-plot, if auto_dim=False.
    col_list: a list with the names of some columns. Only used if auto=False.
    remove_columns: columns to remove to the ones considered if auto=True.
    add_columns:columns to add to the ones considered if auto=True.
    save: if True, the plot will be save as jpg file. If False, not.
    file_name: the name of the jpg file if save=True.
    n_xticks: number of ticks in x-axis.
    fontsize: is the fontsize of the plot tittle.
   
    Returns (outputs)
    ----------
    A histogram matrix of the df data-set, with the parameters specified.
    """

    sns.set_style(style)

    if random == True :
        df = df.sample(fraction=fraction, n=n, seed=seed)

    if response != None and predictors != None : 

        cols_combis = list(product(predictors, response))

    else :

        # Selecting automatically the quantitative columns.
        if auto_col == True :
            quant_col_names = columns_names(df=df, types=[pl.Float64, pl.Int64])
            cat_col_names = columns_names(df=df, types=[pl.Boolean, pl.Utf8])

            if len(remove_quant_col) > 0 :
                for r in remove_quant_col :
                    quant_col_names.remove(r)

            if len(remove_cat_col) > 0 :
                for r in remove_cat_col :
                   cat_col_names.remove(r)
                
            if len(add_quant_col) > 0 : 
                for r in add_quant_col :
                    quant_col_names.append(r)

            if len(add_cat_col) > 0 : 
                for r in add_cat_col :
                    cat_col_names.append(r)

        cols_combis = list(product(cat_col_names, quant_col_names))


    # Define the number of rows and columns for the matrix plot
    n_rows = int(np.ceil(len(cols_combis) / n_cols))

    # Create a subplot with the specified number of rows and columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten()  

    # Defining the colors to be used.
    colors = sns.color_palette("tab20", 20)

    # Defining a ecdf-plot for each variable considered.
    for i, (col1, col2) in enumerate(cols_combis) :
        # If response and predictors not None: col1 = predictors ; col2 = response
        # If response and predictors are None: col1 = cat columns ; col2 = quant columns 

        patches = []
        ax = axes[i]  # Get the current axis
        for value, color in zip(df[col1].unique(), colors):
            Y_cond = df.filter(pl.col(col1)==value).select(col2).to_numpy().flatten()
            sns.ecdfplot(x=Y_cond, color=color, ax=ax, alpha=transparency, complementary=complementary)
            patch = mpatches.Patch(color=color, label=value)
            patches.append(patch)
        ax.set_title(col2 + ' vs ' + col1, fontsize=subtitle_fontsize)
        Y_min = np.floor(df[col2].min())
        Y_max = np.ceil(df[col2].max())        
        xticks_index = np.unique(np.round(np.linspace(Y_min, Y_max, n_xticks)))
        ax.set_xticks(xticks_index)
        ax.set_yticks(np.linspace(0, 1, n_yticks))
        ax.tick_params(axis='x', rotation=x_rotation)
        ax.set_xlabel(col2)
        ax.set_ylabel('Proportion')
        ax.legend(handles=patches, title=col1, loc='upper right', bbox_to_anchor=bbox_to_anchor, fontsize=legend_size)

     # Remove any unused subplots in case the number of 'geo' values is less than num_rows * num_cols
    for j in range(len(cols_combis), n_rows * n_cols):
        fig.delaxes(axes[j])

    # Establishing a general tittle for the plot.
    plt.suptitle(title, fontsize=title_fontsize, y=title_height)
    
    plt.subplots_adjust(hspace=hspace, wspace=wspace) 

    # Setting save options.
    if save == True :         
        fig.savefig(file_name + '.jpg', format='jpg', dpi=400)

    plt.show()  