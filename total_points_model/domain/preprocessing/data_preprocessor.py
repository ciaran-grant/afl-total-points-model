from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, create_team_rolling_averages, rename_team_stats
from total_points_model.domain.contracts.modelling_data_contract import ModellingDataContract
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

from total_points_model.domain.contracts.mappings import Mappings

class DataPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, mapping):
        self.mapping = mapping
        self.ModellingDataContract = ModellingDataContract
        
    def fit(self, X, y=None):
        
        X_copy = X.copy()
        
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        
        X_copy['random5'] = np.random.randint(1, 6, X.shape[0])
        
        X_copy['Round'] = X_copy['Round_ID'].apply(lambda x: x[-2:])

        X_copy = X_copy.replace(self.mapping)

        X_copy = score_col_splitter(X_copy, "Q4_Score")
        
        teams_list = self.ModellingDataContract.team_list
        team_df_list = []
        for team in teams_list:
            X_team = create_team_rolling_averages(X_copy, team)
            team_df_list.append(X_team)
            
        X_team_data = pd.concat(team_df_list).sort_values(by = "Date")
        
        X_home = rename_team_stats(X_team_data, "Home")
        X_copy = pd.merge(X_copy, X_home, how = 'left', on = 'Match_ID')
        X_away = rename_team_stats(X_team_data, "Away")
        X_copy = pd.merge(X_copy, X_away, how = 'left', on = 'Match_ID')

        known_cols_list = ['Home_Team', 'Away_Team', 'Venue', 'Round', 'Year', 'City', 'Temperature', 'Weather_Type', 'random5']
        rolling_cols_list = [x for x in list(X_copy) if "avg" in x]
        modelling_cols = known_cols_list + rolling_cols_list

        X_copy = X_copy[modelling_cols]
        
        # Fitting to training data
        self.train_set_means = X_copy.mean()
        self.weather_mode = X_copy['Weather_Type'].mode()[0]
        self.expected_dummy_cols = list(pd.get_dummies(X_copy))
                
        return self
    
    def transform(self, X):
        
        X['Date'] = pd.to_datetime(X['Date'])
     
        X['random5'] = np.random.randint(1, 6, X.shape[0])

        X['Round'] = X['Round_ID'].apply(lambda x: x[-2:])

        X = X.replace(self.mapping)

        X = score_col_splitter(X, "Q4_Score")
        
        teams_list = self.ModellingDataContract.team_list
        team_df_list = []
        for team in teams_list:
            X_team = create_team_rolling_averages(X, team)
            team_df_list.append(X_team)
            
        X_team_data = pd.concat(team_df_list).sort_values(by = "Date")
        
        X_home = rename_team_stats(X_team_data, "Home")
        X = pd.merge(X, X_home, how = 'left', on = 'Match_ID')
        X_away = rename_team_stats(X_team_data, "Away")
        X = pd.merge(X, X_away, how = 'left', on = 'Match_ID')

        known_cols_list = ['Home_Team', 'Away_Team', 'Venue', 'Round', 'Year', 'City', 'Temperature', 'Weather_Type', 'random5']
        rolling_cols_list = [x for x in list(X) if "avg" in x]
        modelling_cols = known_cols_list + rolling_cols_list

        X = X.sort_values(by = "Date")

        X = X[modelling_cols]
        
        # Applying transformations
        X = X.fillna(self.train_set_means)
        X['Weather_Type'] = X['Weather_Type'].fillna(self.weather_mode)
        
        X = pd.get_dummies(X)
        for col in list(self.expected_dummy_cols):
            if col not in list(X):
                X[col] = 0
                
        X = X[self.expected_dummy_cols]
        

        return X
    