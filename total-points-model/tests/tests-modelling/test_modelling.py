import pandas as pd
import joblib

def get_preprocessed_data():
    
    from total_points_model.config import raw_data_file_path
    from total_points_model.domain.contracts.modelling_data_contract import ModellingDataContract
    from total_points_model.domain.preprocessing.data_preprocessor import DataPreprocessor
    from total_points_model.domain.contracts.mappings import Mappings
    from total_points_model.domain.contracts.rolling_columns import RollingColumns

    afl_data = pd.read_csv(raw_data_file_path)
    afl_data = afl_data[(afl_data['Year'] > 2004) & (afl_data['Year'] < 2023) & ~(afl_data['Year'] == 2020)]
    
    training_data = afl_data[afl_data[ModellingDataContract.TRAIN_TEST_SPLIT_COL]]
    test_data = afl_data[~afl_data[ModellingDataContract.TRAIN_TEST_SPLIT_COL]]

    X_train, y_train = training_data.drop(columns=[ModellingDataContract.RESPONSE]), training_data[ModellingDataContract.RESPONSE]
    X_test, y_test = test_data.drop(columns=[ModellingDataContract.RESPONSE]), test_data[ModellingDataContract.RESPONSE]

    preprocessor = DataPreprocessor(Mappings=Mappings, rolling_dict=RollingColumns.rolling_dict)
    preprocessor.fit(X_train)
    X_train_preproc = preprocessor.transform(X_train)
    X_test_preproc = preprocessor.transform(X_test)
    
    return X_train_preproc, X_test_preproc, y_train, y_test

# Test Hyperparameter Tuning

# Test Year Cross Validation
def test_year_folds():
    
    from total_points_model.domain.modelling.hyperparameter_tuning import XGBYearHyperparameterTuner
    from total_points_model.domain.modelling.optuna_xgb_param_grid import OptunaXGBParamGrid

    X_train_preproc, _, y_train, _ = get_preprocessed_data()
    
    xgb_tuner = XGBYearHyperparameterTuner(X_train_preproc, y_train, optuna_grid=OptunaXGBParamGrid, monotonicity_constraints={})
    
    year_folds_dict = xgb_tuner.define_year_folds()
    
    years_train = (X_train_preproc.iloc[year_folds_dict[2007][0]]['Year']).unique()
    years_valid = (X_train_preproc.iloc[year_folds_dict[2007][1]]['Year']).unique()

    assert (years_train < years_valid).all()

# Test model building
def test_model_build():

    import xgboost as xgb
    from total_points_model.domain.modelling.supermodel import SuperXGBRegressor

    X_train_preproc, X_test_preproc, y_train, y_test = get_preprocessed_data()
    
    params = {
        'max_depth': 17,
        'min_child_weight': 74,
        'eta': 0.018859024830630605,
        'gamma': 2.4446135456353386,
        'lambda': 2.885039290827682,
        'alpha': 7.677219068706761,
        'subsample': 0.7400344429455212,
        'colsample_bytree': 0.7330021414760097,
        'objective': 'reg:squarederror',
        'num_rounds': 1000,
        'early_stopping_rounds': 50,
        'verbosity': 1,
        'monotone_constraints': {}
        }

    super_xgb = SuperXGBRegressor(
        X_train = X_train_preproc, 
        y_train = y_train, 
        X_test = X_test_preproc, 
        y_test = y_test, 
        params = params)
    
    super_xgb.fit()
    
    assert isinstance(super_xgb.xgb_model, xgb.XGBRegressor)
    
# Test monotonicity
def test_monotonicity():
    
    import xgboost as xgb
    from sklearn.inspection import partial_dependence
    from total_points_model.domain.modelling.supermodel import SuperXGBRegressor

    X_train_preproc, X_test_preproc, y_train, y_test = get_preprocessed_data()
    
    params = {
        'max_depth': 17,
        'min_child_weight': 74,
        'eta': 0.018859024830630605,
        'gamma': 2.4446135456353386,
        'lambda': 2.885039290827682,
        'alpha': 7.677219068706761,
        'subsample': 0.7400344429455212,
        'colsample_bytree': 0.7330021414760097,
        'objective': 'reg:squarederror',
        'num_rounds': 1000,
        'early_stopping_rounds': 50,
        'verbosity': 1,
        'monotone_constraints': {'Home_Total_Q4_Score_exp_wavg10':1}
        }

    super_xgb = SuperXGBRegressor(
        X_train = X_train_preproc, 
        y_train = y_train, 
        X_test = X_test_preproc, 
        y_test = y_test, 
        params = params
    )
    super_xgb.fit()
    
    monotone_array = partial_dependence(super_xgb.xgb_model, X_train_preproc.drop(columns=["Match_ID"]), ['Home_Total_Q4_Score_exp_wavg10'])['average']

    assert pd.Series(monotone_array[0]).is_monotonic_increasing or pd.Series(monotone_array[0]).is_monotonic_decreasing
    
    
# Test model predictions
def test_predict():
    
    model_file_path = '/Users/ciaran/Documents/Projects/AFL/total-points-score-model/afl-total-points-model/total_points_model/models/xgb_total_points_v8.joblib'
    xgb_model = joblib.load(model_file_path)
    
    X = pd.DataFrame({
        'Year': {2754: 2018.0},
        'Round': {2754: 28},
        'Finals': {2754: True},
        'Primary_Home': {2754: False},
        'Home_Team_Within_State': {2754: False},
        'Away_Team_Within_State': {2754: True},
        'ELO_Home': {2754: 1653.2474186767956},
        'ELO_Away': {2754: 1591.2895905932278},
        'ELO_diff': {2754: 61.95782808356785},
        'ELO_abs_diff': {2754: 61.95782808356785},
        'ELO_probs_Home': {2754: 0.588231172688287},
        'ELO_probs_Away': {2754: 0.41176882731171305},
        'ELO_probs_diff': {2754: 0.1764623453765739},
        'ELO_probs_abs_diff': {2754: 0.1764623453765739},
        'Home_Total_Q4_Score_exp_wavg10': {2754: 166.14513823188324},
        'Away_Total_Q4_Score_exp_wavg10': {2754: 157.31893428952324},
        'Home_Total_Q4_Shots_exp_wavg10': {2754: 45.508000684304925},
        'Away_Total_Q4_Shots_exp_wavg10': {2754: 43.05940027292787},
        'Kicking_Weather_Bad Kicking': {2754: 1},
        'Kicking_Weather_Good Kicking': {2754: 0},
        'State_International': {2754: 0},
        'State_New South Wales': {2754: 0},
        'State_Northern Territory': {2754: 0},
        'State_Queensland': {2754: 0},
        'State_South Australia': {2754: 0},
        'State_Tasmania': {2754: 0},
        'State_Victoria': {2754: 1},
        'State_Wellington': {2754: 0},
        'State_Western Australia': {2754: 0},
        'Victoria_Not Victoria': {2754: 0},
        'Victoria_Victoria': {2754: 1},
        'Victoria_Wellington': {2754: 0},
        'Roof_Blacktown': {2754: 0},
        'Roof_Football Park': {2754: 0},
        'Roof_No Roof': {2754: 1},
        'Roof_Princes Park': {2754: 0},
        'Roof_Roof': {2754: 0},
        'Roof_Subiaco': {2754: 0},
        'Roof_Wellington': {2754: 0}
        })
    
    pred = xgb_model.predict(X)
    
    assert pred == 152.49583

