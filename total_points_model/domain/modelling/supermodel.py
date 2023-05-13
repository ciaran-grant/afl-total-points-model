import xgboost as xgb
import joblib

class SuperModel:
    
    def __init__(self, X_train, y_train, X_test, y_test, params):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = params
    
class SuperXGBRegressor(SuperModel):
    def __init__(self, X_train, y_train, X_test, y_test, params):
        super().__init__(X_train, y_train, X_test, y_test, params)
        
        self.xgb_params = self._get_xgb_hyperparameters()
    
    def _get_xgb_hyperparameters(self):
        
        xgb_params = {
            'max_depth': self.params['max_depth'],
            'min_child_weight': self.params['min_child_weight'],
            'eta': self.params['eta'],
            'gamma': self.params['gamma'],
            'lambda': self.params['lambda'],
            'alpha': self.params['alpha'],
            'subsample': self.params['subsample'],
            'colsample_bytree': self.params['colsample_bytree']
        }
        
        return xgb_params
    
    def fit(self):
                
        self.xgb_reg = xgb.XGBRegressor(n_estimators=self.params['num_rounds'],
                                        objective = self.params['objective'],
                                        verbosity = self.params['verbosity'],
                                        early_stopping_rounds = self.params['early_stopping_rounds'],
                                        learning_rate = self.params['eta'],
                                        max_depth = self.params['max_depth'],
                                        min_child_weight = self.params['min_child_weight'],
                                        gamma = self.params['gamma'],
                                        subsample = self.params['subsample'],
                                        colsample_bytree = self.params['colsample_bytree'],
                                        reg_alpha = self.params['alpha'],
                                        reg_lambda = self.params['lambda']

                                        )
        
        self.xgb_model = self.xgb_reg.fit(X = self.X_train,
                                          y = self.y_train,
                                          eval_set = [(self.X_train, self.y_train), (self.X_test, self.y_test)])
        
    def predict(self, X):
        
        # dmatrix = xgb.DMatrix(X)

        return self.xgb_model.predict(X)
    
    def export_model(self, file_path):
        
        joblib.dump(self.xgb_model, file_path)