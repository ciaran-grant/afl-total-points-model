import optuna
import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from total_points_model.domain.modelling.optuna_xgb_param_grid import OptunaXGBParamGrid
from total_points_model.domain.contracts.modelling_data_contract import ModellingDataContract

class HyperparameterTuner:
    
    def __init__(self, training_data, response):
        self.training_data = training_data
        self.response = response
             
class XGBHyperparameterTuner(HyperparameterTuner, OptunaXGBParamGrid):
    
    def __init__(self, training_data, response) -> None:
        super().__init__(training_data, response)
        
    def objective(self, trial):

        train_x, valid_x, train_y, valid_y = train_test_split(self.training_data, self.response, test_size=self.validation_size)
        dtrain = xgb.DMatrix(train_x, label=train_y)
        dvalid = xgb.DMatrix(valid_x, label=valid_y)

        param = {
            "verbosity": self.verbosity,
            'objective': self.error,
            # maximum depth of the tree, signifies complexity of the tree.
            "max_depth" : trial.suggest_int("max_depth",
                                            self.max_depth_min,
                                            self.max_depth_max,
                                            step=self.max_depth_step),
            # minimum child weight, larger the term more conservative the tree.
            "min_child_weight" : trial.suggest_int("min_child_weight", 
                                                self.min_child_weight_min,
                                                self.min_child_weight_max,
                                                step=self.min_child_weight_step),
            "eta" : trial.suggest_float("eta",
                                        self.eta_min, 
                                        self.eta_max, 
                                        log=True),
            # defines how selective algorithm is.
            "gamma" : trial.suggest_float("gamma", 
                                        self.min_child_weight_max, 
                                        self.min_child_weight_max, 
                                        log=True),
            # L2 regularization weight.
            "lambda": trial.suggest_float("lambda",
                                        self.lambda_min,
                                        self.lambda_max, 
                                        log=True),
            # L1 regularization weight.
            "alpha": trial.suggest_float("alpha", 
                                        self.alpha_min,
                                        self.alpha_max,
                                        log=True),
            # sampling ratio for training data.
            "subsample": trial.suggest_float("subsample", 
                                            self.subsample_min, 
                                            self.subsample_max),
            # sampling according to each tree.
            "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                    self.colsample_bytree_min, 
                                                    self.colsample_bytree_max),
        }        

        bst = xgb.train(param, dtrain)
        preds = bst.predict(dvalid)
        
        rmse = mean_squared_error(preds, valid_y, squared=False)
        
        return rmse

        
    def get_objective_function(self):
        return self.objective
    
    def tune_hyperparameters(self):
    
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=self.trials)
        
        print("Number of finished trials: ", len(self.study.trials))
        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        return self.study
    
    def get_best_params(self):
        return self.study.best_params
            
             
class XGBYearHyperparameterTuner(HyperparameterTuner): 
        
    def __init__(self, training_data, response, optuna_grid, monotonicity_constraints):
        super().__init__(training_data, response)
        self.optuna_grid = optuna_grid
        self.monotonicity_constraints = monotonicity_constraints
    
    def define_year_folds(self):
    
        years_list = list(self.training_data['Year'].unique())
        test_years = years_list[1:]
        
        year_folds_dict = {}
        for year in test_years:
            (train_fold, test_fold) = self.training_data.loc[self.training_data['Year'] < year].drop(columns = ['Year']), self.training_data[self.training_data['Year'] == year].drop(columns = ['Year'])
            year_folds_dict[year] = (train_fold.index, test_fold.index)
        
        return year_folds_dict
    
    def objective(self, trial):

        year_folds_dict = self.define_year_folds()

        validation_error = []
        
        for year, fold in year_folds_dict.items():

            train_x, valid_x, train_y, valid_y = self.training_data.iloc[fold[0]], self.training_data.iloc[fold[1]], self.response.iloc[fold[0]], self.response.iloc[fold[1]]
            
            train_x_features = train_x.drop(columns = [ModellingDataContract.ID_COL])
            valid_x_features = valid_x.drop(columns = [ModellingDataContract.ID_COL])
            
            dtrain = xgb.DMatrix(train_x_features, label=train_y)
            dvalid = xgb.DMatrix(valid_x_features, label=valid_y)

            param = {
                "verbosity": self.optuna_grid.verbosity,
                'objective': self.optuna_grid.error,
                # maximum depth of the tree, signifies complexity of the tree.
                "max_depth" : trial.suggest_int("max_depth",
                                                self.optuna_grid.max_depth_min,
                                                self.optuna_grid.max_depth_max,
                                                step=self.optuna_grid.max_depth_step),
                # minimum child weight, larger the term more conservative the tree.
                "min_child_weight" : trial.suggest_int("min_child_weight", 
                                                    self.optuna_grid.min_child_weight_min,
                                                    self.optuna_grid.min_child_weight_max,
                                                    step=self.optuna_grid.min_child_weight_step),
                "eta" : trial.suggest_float("eta",
                                            self.optuna_grid.eta_min, 
                                            self.optuna_grid.eta_max
                                            ),
                # defines how selective algorithm is.
                "gamma" : trial.suggest_float("gamma", 
                                            self.optuna_grid.gamma_min, 
                                            self.optuna_grid.gamma_max
                                            ),
                # L2 regularization weight.
                "lambda": trial.suggest_float("lambda",
                                            self.optuna_grid.lambda_min,
                                            self.optuna_grid.lambda_max
                                            ),
                # L1 regularization weight.
                "alpha": trial.suggest_float("alpha", 
                                            self.optuna_grid.alpha_min,
                                            self.optuna_grid.alpha_max
                                            ),
                # sampling ratio for training data.
                "subsample": trial.suggest_float("subsample", 
                                                self.optuna_grid.subsample_min, 
                                                self.optuna_grid.subsample_max),
                # sampling according to each tree.
                "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                        self.optuna_grid.colsample_bytree_min, 
                                                        self.optuna_grid.colsample_bytree_max),
            }  
            param['monotone_constraints'] = self.monotonicity_constraints

            bst = xgb.train(param,
                            dtrain,
                            num_boost_round = self.optuna_grid.num_rounds,
                            evals = [(dtrain, "train"), (dvalid, "valid")],
                            early_stopping_rounds = self.optuna_grid.early_stopping_rounds)
            preds = bst.predict(dvalid)
        
            rmse = mean_squared_error(preds, valid_y, squared=False)
            
            validation_error.append(rmse)

        return np.mean(validation_error)

        
    def get_objective_function(self):
        return self.objective
    
    def tune_hyperparameters(self):
    
        self.study = optuna.create_study(pruner = optuna.pruners.MedianPruner(), direction='minimize')
        self.study.optimize(self.objective, n_trials=self.optuna_grid.trials)
        
        print("Number of finished trials: ", len(self.study.trials))
        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: {}".format(trial.value))
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
            
        return self.study
    
    def get_best_params(self):
        return self.study.best_params
