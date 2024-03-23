################################################################################

import numpy as np
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, PredefinedSplit, GridSearchCV
import polars as pl
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, KBinsDiscretizer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectKBest, SelectPercentile, f_regression, f_classif, mutual_info_classif, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna

################################################################################

class OptunaSearchCV:
    
    def __init__(self, estimator, param_grid, cv, scoring, direction='minimize', n_iter=10, random_state=123):

        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.direction = direction
        self.n_iter = n_iter
        self.random_state = random_state

    def objective(self, trial, X, y):
 
       self.estimator.set_params(**self.param_grid(trial))
       score = np.mean(cross_val_score(X=X, y=y, estimator=self.estimator, scoring=self.scoring, cv=self.cv))
       return score 
    
    def fit(self, X, y):
       
       sampler = optuna.samplers.TPESampler(seed=self.random_state)
       study = optuna.create_study(direction=self.direction, sampler=sampler)
       study.optimize(lambda trial: self.objective(trial, X=X, y=y), n_trials=self.n_iter) 
       self.best_params_ = study.best_params
       self.best_score_ = study.best_value 
       self.study = study

    def results(self):
          # Collect trial information
        results = []
        for trial in self.study.trials:
            trial_data = {
              'params': trial.params,
              'score': trial.value,
              }
            results.append(trial_data)

        # Create a DataFrame from the collected information
        results = pd.DataFrame(results)
        results = pd.concat((results['params'].apply(lambda x: pd.Series(x)), results['score']), axis=1)
        if 'neg' in self.scoring:
            results['score'] = - results['score']
            results = results.sort_values(by='score', ascending=True)
        else:
            results = results.sort_values(by='score', ascending=False)
        return results
    
################################################################################

def optuna_nested_results(search, estimator, inner_results, outer_scores, scoring, X_train, X_test, Y_train, Y_test):

    # Inner results (HPO)
    inner_results.append(search.results())
    # Outer score (estimation future performance)
    best_estimator = estimator.set_params(**search.best_params_)
    best_estimator.fit(X_train, Y_train)
    Y_test_hat = best_estimator.predict(X_test)
    if scoring == 'neg_mean_absolute_error':
        outer_scores.append(mean_absolute_error(y_true=Y_test, y_pred=Y_test_hat))
    elif scoring == 'neg_mean_squared_error':
        outer_scores.append(mean_squared_error(y_true=Y_test, y_pred=Y_test_hat))   

    return inner_results, outer_scores

################################################################################

def format_results(search, scoring):

    results = pl.DataFrame(search.cv_results_)
    columns_to_keep = [x for x in results.columns if 'param' in x and x != 'params'] + ['mean_test_score']
    results = results[columns_to_keep]
    if 'neg' in scoring:
        results = results.with_columns(-pl.col('mean_test_score'))
        results = results.sort(by='mean_test_score', descending=False)
    else:
        results = results.sort(by='mean_test_score', descending=True)
    rename_columns = ['_'.join(x.split('_')[1:]) for x in columns_to_keep[:(len(columns_to_keep)-1)]]
    rename_columns = rename_columns + ['score']
    results.columns = rename_columns
    return results

################################################################################

# A function that is used in SemiNested Evaluation
def predefine_split(data_size=None, train_prop=None, random=None, random_state=None, train_indices=None, test_indices=None):
    
    if None in [train_indices, test_indices]:
        train_size = round(train_prop*data_size)
        test_size = data_size - train_size
        train_indices = np.repeat(-1, train_size) # -1 = Train 
        test_indices = np.repeat(0, test_size)  # 0 = Test
        indices = np.concatenate((train_indices, test_indices))
        if random == True:
            np.random.seed(random_state)
            indices = np.random.choice(indices, len(indices), replace=False)
    else:
        train_size = len(train_indices) ; test_size = len(test_indices)
        indices = np.zeros(train_size + test_size)
        indices[train_indices] = -1
    
    return PredefinedSplit(indices)

# Examples of usage
# predefine_split(data_size=len(X_train), train_prop=0.75, random=True, random_state=123)
# predefine_split(data_size=len(X), train_prop=0.75, random=False)
# predefine_split(train_indices=[0,2,3,5,10], test_indices=[1,4,6,7,8,9])

################################################################################

class SimpleEvaluation: # Outer: Simple Validation ; Inner: Simple or CV

    def __init__(self, estimator, param_grid, inner, search_method, scoring, direction='minimize', n_trials=10, random_state=123):
            
        self.estimator = estimator
        self.param_grid = param_grid
        self.inner = inner
        self.search_method = search_method
        self.scoring = scoring
        self.direction = direction
        self.n_trials = n_trials
        self.random_state = random_state
    
    def fit(self, X, Y):
            
        # Inner HPO
        if self.search_method == 'optuna':

            search = OptunaSearchCV(estimator=self.estimator, param_grid=self.param_grid,       
                                    cv=self.inner, scoring=self.scoring, direction=self.direction,
                                    n_iter=self.n_trials, random_state=self.random_state)
            
        else:

            if self.search_method == 'random_search':              
                search = RandomizedSearchCV(estimator=self.estimator, param_distributions=self.param_grid, 
                                            cv=self.inner, scoring=self.scoring, 
                                            n_iter=self.n_trials, random_state=self.random_state)

            elif self.search_method == 'grid_search':              

                search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, 
                                      cv=self.inner, scoring=self.scoring)          
            
        search.fit(X, Y)
        # Inner results (HPO)
        self.inner_results = search.results() if self.search_method == 'optuna' else format_results(search, self.scoring)

        # Inner results (HPO)    
        self.inner_best_params = search.best_params_
        if 'neg' in self.scoring:
            self.inner_score = - search.best_score_  
        else:
            self.inner_score = search.best_score_

################################################################################

class NestedEvaluation: # Outer: CV ; Inner: CV

    def __init__(self, estimator,  inner, outer, scoring, direction='minimize', param_grid=None, search_method=None, n_trials=10, random_state=123):
         
        self.estimator = estimator
        self.param_grid = param_grid
        self.inner = inner
        self.outer = outer
        self.search_method = search_method
        self.scoring = scoring
        self.direction = direction
        self.n_trials = n_trials  
        self.random_state = random_state      

    def fit(self, X, Y):

        inner_scores, outer_scores, inner_best_params, inner_results = [], [], [], []

        for k, (train_index, test_index) in enumerate(self.outer.split(X)) :
            
            print('----------------')
            print(f'Outer: Fold {k+1}')
            print('----------------')
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            elif isinstance(X, np.ndarray):
                X_train, X_test = X[train_index,:], X[test_index,:]
            Y_train, Y_test = Y[train_index], Y[test_index]

            if self.search_method is None:
                
                inner_score = np.mean(cross_val_score(estimator=self.estimator, X=X_train, y=Y_train, scoring=self.scoring, cv=self.inner))
                inner_score = -inner_score if 'neg' in self.scoring else inner_score
                inner_scores.append(inner_score)
                self.estimator.fit(X=X_train, y=Y_train)
                Y_test_hat = self.estimator.predict(X_test)
                if self.scoring == 'neg_mean_absolute_error':
                    outer_scores.append(mean_absolute_error(y_true=Y_test, y_pred=Y_test_hat))
                elif self.scoring == 'neg_mean_squared_error':
                    outer_scores.append(mean_squared_error(y_true=Y_test, y_pred=Y_test_hat)) 

            else:

                # Inner HPO
                if self.search_method == 'optuna':

                    search = OptunaSearchCV(estimator=self.estimator, param_grid=self.param_grid, 
                        cv=self.inner, scoring=self.scoring, direction=self.direction, n_iter=self.n_trials, seed=self.random_state)
                    search.fit(X_train, Y_train)
                    
                    inner_results, outer_scores = optuna_nested_results(search, self.estimator, inner_results, outer_scores, self.scoring, X_train, X_test, Y_train, Y_test)

                else:
                    
                    if self.search_method == 'random_search':
                
                        search = RandomizedSearchCV(estimator=self.estimator, param_distributions=self.param_grid, 
                                cv=self.inner, scoring=self.scoring, n_iter=self.n_trials, random_state=self.random_state)
                        search.fit(X_train, Y_train)

                    elif self.search_method == 'grid_search':              

                        search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, 
                                                cv=self.inner, scoring=self.scoring)  
                    
                    # Inner results (HPO)
                    inner_results.append(format_results(search, self.scoring))
                    # Outer score (estimation future performance)
                    outer_scores.append(-search.score(X=X_test, y=Y_test))                   
            
                # Inner (HPO) best_params and score
                inner_best_params.append(search.best_params_)
                inner_scores.append(-search.best_score_) if 'neg' in self.scoring else inner_scores.append(search.best_score_)

        self.inner_results = inner_results
        self.inner_best_params = inner_best_params
        self.outer_scores = np.array(outer_scores)
        self.inner_scores = np.array(inner_scores)
        self.final_inner_score = np.mean(self.inner_scores)
        self.final_outer_score = np.mean(self.outer_scores) # Estimation of future performance
        # The one with the least MAE. This is a criteria to obtain the finals params, but not the only possible.
        if self.search_method is not None:
            self.final_best_params = inner_best_params[np.argmin(self.inner_scores)]

################################################################################

class SemiNestedEvaluation: # Outer: CV ; Inner: Simple

    def __init__(self, estimator, param_grid, outer, search_method, scoring, direction='maximize', n_trials=10, 
                 random_state=123, train_prop=0.75, random_sv=True, train_indices=None, test_indices=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.outer = outer
        self.search_method = search_method
        self.scoring = scoring
        self.direction = direction
        self.n_trials = n_trials
        self.random_state = random_state
        self.train_prop = train_prop
        self.random_sv = random_sv
        self.train_indices = train_indices
        self.test_indices = test_indices
    
    def fit(self, X, Y):

        inner_scores, outer_scores, inner_best_params, inner_results = [], [], [], []

        for k, (train_index, test_index) in enumerate(self.outer.split(X)) :
            
            print('----------------')
            print(f'Outer: Fold {k+1}')
            print('----------------')
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
            elif isinstance(X, np.ndarray):
                X_train, X_test = X[train_index,:], X[test_index,:]
            Y_train, Y_test = Y[train_index], Y[test_index]

            # Inner definition (differential step)
            self.inner = predefine_split(data_size=len(X_train), train_prop=self.train_prop, random=self.random_sv, 
                                    random_state=self.random_state, train_indices=self.train_indices, test_indices=self.test_indices)
            # Inner HPO
            if self.search_method == 'optuna':

                search = OptunaSearchCV(estimator=self.estimator, param_grid=self.param_grid, 
                    cv=self.inner, scoring=self.scoring, direction=self.direction, n_iter=self.n_trials, seed=self.random_state)
                search.fit(X_train, Y_train)
                
                inner_results, outer_scores = optuna_nested_results(search, self.estimator, inner_results, outer_scores, self.scoring, X_train, X_test, Y_train, Y_test)

            else:
                
                if self.search_method == 'random_search':
            
                    search = RandomizedSearchCV(estimator=self.estimator, param_distributions=self.param_grid, 
                            cv=self.inner, scoring=self.scoring, n_iter=self.n_trials, random_state=self.random_state)
                    search.fit(X_train, Y_train)

                elif self.search_method == 'grid_search':              

                    search = GridSearchCV(estimator=self.estimator, param_grid=self.param_grid, 
                                            cv=self.inner, scoring=self.scoring)  
                
                # Inner results (HPO)
                inner_results.append(format_results(search, self.scoring))
                # Outer score (estimation future performance)
                outer_scores.append(-search.score(X=X_test, y=Y_test))                   
        
            # Inner (HPO) best_params and score
            inner_best_params.append(search.best_params_)
            inner_scores.append(-search.best_score_) if 'neg' in self.scoring else inner_scores.append(search.best_score_)

        self.inner_results = inner_results
        self.inner_best_params = inner_best_params
        self.outer_scores = np.array(outer_scores)
        self.inner_scores = np.array(inner_scores)
        self.final_inner_score = np.mean(self.inner_scores)
        self.final_outer_score = np.mean(self.outer_scores) # Estimation of future performance
        # The one with the least MAE. This is a criteria to obtain the finals params, but not the only possible.
        self.final_best_params = inner_best_params[np.argmin(self.inner_scores)]

################################################################################

class encoder(BaseEstimator, TransformerMixin):

    def __init__(self, method='ordinal', drop='first', handle_unknown='error'): # drop=None to not remove any dummy
        self.method = method
        self.drop = drop
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):

        if self.method == 'ordinal':
            self.encoder_ = OrdinalEncoder()
        elif self.method == 'one-hot':
            self.encoder_ = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=True)
        else:
            raise ValueError("Invalid method for encoding")
        
        self.encoder_.fit(X)
        return self

    def transform(self, X):
        
        if self.method == 'one-hot':
            # One-hot encoding gives an sparse matrix as output.
            # The output is transformed from sparse to dense matrix since this is usually required in sklearn.
            X = self.encoder_.transform(X).toarray() 
        else: 
            X = self.encoder_.transform(X)


        return X
    
################################################################################
       
class imputer(BaseEstimator, TransformerMixin):

    def __init__(self, apply=True, method='simple_median', n_neighbors=1, n_nearest_features=4):
        self.apply = apply
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_nearest_features = n_nearest_features

    def fit(self, X, y=None):

        if self.apply == True:

            if self.method in ['simple_mean', 'simple_median', 'simple_most_frequent']:
                 self.imputer_ = SimpleImputer(missing_values=np.nan, strategy='_'.join(self.method.split('_')[1:]))
            elif self.method == 'knn':
                 self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors, weights="uniform")
            elif self.method in ['iterative_mean', 'iterative_median', 'iterative_most_frequent']:
                 self.imputer_ = IterativeImputer(initial_strategy='_'.join(self.method.split('_')[1:]), 
                                                  n_nearest_features=self.n_nearest_features, max_iter=25, random_state=123)
            else:
                 raise ValueError("Invalid method for imputation")
           
            self.imputer_.fit(X)
        return self

    def transform(self, X):
        
        if self.apply == True:
            X = self.imputer_.transform(X) # Output: numpy array
        return X
    
################################################################################

class scaler(BaseEstimator, TransformerMixin):

    def __init__(self, apply=False, method='standard'):
        self.apply = apply
        self.method = method

    def fit(self, X, y=None):
        
        if self.apply == True:
            if self.method == 'standard':
                self.scaler_ = StandardScaler(with_mean=True, with_std=True)
            elif self.method == 'min-max':
                self.scaler_ = MinMaxScaler(feature_range=(0, 1))

            self.scaler_.fit(X)
        return self
    
    def transform(self, X):
        
        if self.apply == True:
            X = self.scaler_.transform(X)
        return X        
    
################################################################################

class clip_response(BaseEstimator, TransformerMixin):

    def __init__(self, apply=False):
        self.apply = apply

    def fit(self, X, y=None):        
        return self
    
    def transform(self, X):
        # Clip values between 0 and 1
        clipped_values = np.clip(X, 0, 1)
        return clipped_values

################################################################################
class discretizer(BaseEstimator, TransformerMixin):

    def __init__(self, apply=False, n_bins=3, strategy='quantile'):
        self.apply = apply
        self.n_bins = n_bins
        self.strategy = strategy

    def fit(self, X, y=None):
        
        if self.apply == True:
            self.discretizer_ = KBinsDiscretizer(encode='ordinal', n_bins=self.n_bins, strategy=self.strategy)
            self.discretizer_.fit(X)
        return self
    
    def transform(self, X):
        
        if self.apply == True:
            X = self.discretizer_.transform(X)
        return X  
    
################################################################################

class features_selector(BaseEstimator, TransformerMixin):

    def __init__(self, apply=False, method='Fdr', cv=3, k=5, percentile=30, n_jobs=None):
        self.apply = apply
        self.method = method
        self.cv = cv
        self.k = k
        self.percentile = percentile
        self.n_jobs = n_jobs

    def fit(self, X, y):
        
        if self.apply == True:

            if self.method == 'Fdr_reg':
                self.features_selector_ = SelectFdr(f_regression, alpha=0.05)
            elif self.method == 'Fpr_reg':
                self.features_selector_ = SelectFpr(f_regression, alpha=0.05)
            elif self.method == 'Fdr_f_class':
                self.features_selector_ = SelectFdr(f_classif, alpha=0.05)
            elif self.method == 'Fpr_f_class':
                self.features_selector_ = SelectFpr(f_classif, alpha=0.05)
            elif self.method == 'KBest_mutual_class':
                self.features_selector_ = SelectKBest(mutual_info_classif, k=self.k)
            elif self.method == 'Percentile_mutual_class':
                self.features_selector_ = SelectPercentile(mutual_info_classif, percentile=20)

            elif self.method == 'forward_linear_regression':
                self.features_selector_ = SequentialFeatureSelector(estimator=LinearRegression(),
                                                                    n_features_to_select='auto',
                                                                    direction='forward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'backward_linear_regression':
                self.features_selector_ = SequentialFeatureSelector(estimator=LinearRegression(),
                                                                    n_features_to_select='auto',
                                                                    direction='backward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'forward_knn_reg':
                self.features_selector_ = SequentialFeatureSelector(estimator=KNeighborsRegressor(n_neighbors=5),
                                                                    n_features_to_select='auto',
                                                                    direction='forward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'backward_knn_reg':
                self.features_selector_ = SequentialFeatureSelector(estimator=KNeighborsRegressor(n_neighbors=5),
                                                                    n_features_to_select='auto',
                                                                    direction='backward', cv=self.cv, n_jobs=self.n_jobs) 
            elif self.method == 'forward_knn_class':
                self.features_selector_ = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=5),
                                                                    n_features_to_select='auto',
                                                                    direction='forward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'backward_knn_class':
                self.features_selector_ = SequentialFeatureSelector(estimator=KNeighborsClassifier(n_neighbors=5),
                                                                    n_features_to_select='auto',
                                                                    direction='backward', cv=self.cv, n_jobs=self.n_jobs) 
            elif self.method == 'forward_logistic_regression':
                self.features_selector_ = SequentialFeatureSelector(estimator=LogisticRegression(),
                                                                    n_features_to_select='auto',
                                                                    direction='forward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'backward_logistic_regression':
                self.features_selector_ = SequentialFeatureSelector(estimator=LogisticRegression(),
                                                                    n_features_to_select='auto',
                                                                    direction='backward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'backward_trees_class':
                self.features_selector_ = SequentialFeatureSelector(estimator=DecisionTreeClassifier(max_depth=4),
                                                                    n_features_to_select='auto',
                                                                    direction='backward', cv=self.cv, n_jobs=self.n_jobs)
            elif self.method == 'forward_trees_class':
                self.features_selector_ = SequentialFeatureSelector(estimator=DecisionTreeClassifier(max_depth=4),
                                                                    n_features_to_select='auto',
                                                                    direction='forward', cv=self.cv, n_jobs=self.n_jobs)
            else:
                raise ValueError("Invalid method for features selector")
        
            self.features_selector_.fit(X, y)
        return self
    
    def transform(self, X):
        
        if self.apply == True:
            X = self.features_selector_.transform(X)
        return X  
    
################################################################################

def absolute_r2_score(y_true, y_pred):

    Y_test_hat_mean_model = np.repeat(np.mean(y_true), len(y_true))
    MAE_best_model = mean_absolute_error(y_pred=y_pred, y_true=y_true)
    MAE_mean_model = mean_absolute_error(y_pred=Y_test_hat_mean_model, y_true=y_true)
    absolute_r2_score = 1 - (MAE_best_model / MAE_mean_model)
    
    return absolute_r2_score

################################################################################

def predictive_plots(Y, Y_hat, model_name, future_performance, n_random_samples):

    if isinstance(Y, (pd.Series, pl.Series)):
        Y = Y.to_numpy()

    # Calculate residuals and percentage deviation
    percentage_deviation = (np.abs(Y - Y_hat) / (Y - np.mean(Y_hat))) * 100
    # Create an array of x-axis values
    x_values = np.arange(len(Y))
    print(f'Estimation of future performance: \nMAE = {np.round(future_performance, 3)}')
    print(f'Absolute R2 = {np.round(absolute_r2_score(y_pred=Y_hat, y_true=Y), 3) * 100} %')

    fig, axes = plt.subplots(figsize=(15, 6))
    # Plot 1: True values vs. Predicted values
    ax = sns.scatterplot(y=Y_hat, x=Y, label='Predicted vs True', color='blue')
    ax = sns.regplot(x=Y, y=Y, scatter=False, 
                line_kws={'linestyle':'--', 'color':'red'}, 
                label='Perfect Fit')
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()
    ax.set_title(f'Predicted vs True Values - {model_name}', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    # Plot 2: Line plot of True vs. Predicted values
    ax = sns.lineplot(x=x_values, y=Y, color='red', label='True Values')
    ax = sns.lineplot(x=x_values, y=Y_hat, color='blue', label='Predicted Values')
    ax.legend()
    ax.set_xlabel('Data Points')
    ax.set_ylabel('Values')
    ax.set_title(f'True vs Predicted Values - {model_name}', fontsize=15)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 5))
    # Plot 3: Line plot of True vs. Predicted values
    np.random.seed(123)
    x_values_random = np.random.choice(x_values, n_random_samples, replace=False)
    ax = sns.lineplot(x=x_values_random, y=Y[x_values_random], color='red', label='True Values', marker='o', markersize=5)
    ax = sns.lineplot(x=x_values_random, y=Y_hat[x_values_random], color='blue', label='Predicted Values', marker='o', markersize=5)
    ax.legend()
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.set_title(f'True vs Predicted Values ({n_random_samples} random data points I)  - {model_name}', fontsize=15)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 5))
    # Plot 4: Line plot of True vs. Predicted values
    np.random.seed(100)
    x_values_random = np.random.choice(x_values, n_random_samples, replace=False)
    ax = sns.lineplot(x=x_values_random, y=Y[x_values_random], color='red', label='True Values', marker='o', markersize=5)
    ax = sns.lineplot(x=x_values_random, y=Y_hat[x_values_random], color='blue', label='Predicted Values', marker='o', markersize=5)
    ax.legend()
    ax.set_xlabel('Index')
    ax.set_ylabel('Values')
    ax.set_title(f'True vs. Predicted Values ({n_random_samples} random points II)  - {model_name}', fontsize=15)
    plt.tight_layout()
    plt.show()

################################################################################
    
def predictive_intervals(estimator, B, X_train, Y_train, X_test, Y_test, Y_test_hat, n_points=50, random_state=123, model_name=None):

   Y_test_hat_b = np.zeros((len(Y_test),B))
   for j, b in enumerate(range(0, B)):
      np.random.seed(random_state + b)
      bootstrap_indices = np.random.choice(np.arange(0,len(Y_train)), len(Y_train), replace=True)
      X_train_b, Y_train_b = X_train.iloc[bootstrap_indices,:], Y_train.iloc[bootstrap_indices]
      try:
         estimator.fit(X_train_b, Y_train_b)
         Y_test_hat_b[:,j] = estimator.predict(X_test)
      except:
         pass

   sd_bootstrap = np.std(Y_test_hat_b, axis=1)
   PI_1 = Y_test_hat - sd_bootstrap
   PI_2 = Y_test_hat + sd_bootstrap

   fig, axes = plt.subplots(figsize=(18, 6))
   np.random.seed(random_state)
   random_data_indices = np.random.choice(np.arange(0, len(X_test)), n_points, replace=False)

   # Scatter plot for actual data points
   sns.scatterplot(x=random_data_indices, y=Y_test.iloc[random_data_indices], 
                   color='red', label='True values')

   # Line plot for predicted values
   sns.lineplot(x=random_data_indices, y=Y_test_hat[random_data_indices], 
                color='blue', label='predicted values', marker='o', markersize=5)

   # Add a shaded area for the predictive intervals
   plt.fill_between(np.sort(random_data_indices), PI_1[np.sort(random_data_indices)], PI_2[np.sort(random_data_indices)],
                    color='green', alpha=0.3)
   #sns.lineplot(x=random_data_indices, y=PI_1[random_data_indices], color='fuchsia', linestyle='--')
   #sns.lineplot(x=random_data_indices, y=PI_2[random_data_indices], color='orange', linestyle='--')
   plt.set_title(f'Prediction intervals and Predicted values vs. True Values ({n_points} random data points) - {model_name}', fontsize=15)
   plt.show()

   print(f'The percentage of true values within the prediction intervals is {np.mean([(x >= PI_1) and (x <= PI_2) for x in Y_test_hat[random_data_indices]])}')

################################################################################
    


################################################################################
    

################################################################################