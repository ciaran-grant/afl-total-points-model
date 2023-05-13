from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, create_team_rolling_averages, rename_team_stats
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

class DataPreprocessor(BaseEstimator, TransformerMixin):
    
    def __init__(self, rename_dict):
        self.rename_dict = rename_dict
     
    def fit(self, X, y=None):
        
        X_copy = X.copy()
        
        X_copy = X_copy.rename(columns=self.rename_dict)
        
        X_copy['random5'] = np.random.randint(1, 6, X.shape[0])
        
        X_copy = score_col_splitter(X_copy, "Q1_Score")
        X_copy = score_col_splitter(X_copy, "Q2_Score")
        X_copy = score_col_splitter(X_copy, "Q3_Score")
        X_copy = score_col_splitter(X_copy, "Q4_Score")
        
        teams_list = list(X_copy['Home_Team'].unique())
        team_df_list = []
        for team in teams_list:
            X_team = create_team_rolling_averages(X_copy, team)
            team_df_list.append(X_team)
            
        X_team_data = pd.concat(team_df_list).sort_values(by = "Date")
        
        X_home = rename_team_stats(X_team_data, "Home")
        X_copy = pd.merge(X_copy, X_home, how = 'left', on = 'Match_ID')
        X_away = rename_team_stats(X_team_data, "Away")
        X_copy = pd.merge(X_copy, X_away, how = 'left', on = 'Match_ID')

        known_cols_list = ['Home_Team', 'Away_Team', 'Venue', 'Round_ID', 'Year', 'City', 'Temperature', 'Weather_Type', 'random5']
        rolling_cols_list = [x for x in list(X_copy) if "avg" in x]
        modelling_cols = known_cols_list + rolling_cols_list

        X_copy = X_copy[modelling_cols]
        
        # Fitting to training data
        self.train_set_means = X_copy.mean()
        self.weather_mode = X_copy['Weather_Type'].mode()[0]
        self.expected_dummy_cols = list(pd.get_dummies(X_copy))
                
        return self
    
    def transform(self, X):
        
        X = X.rename(columns=self.rename_dict)
        
        X['random5'] = np.random.randint(1, 6, X.shape[0])
        
        X = score_col_splitter(X, "Q1_Score")
        X = score_col_splitter(X, "Q2_Score")
        X = score_col_splitter(X, "Q3_Score")
        X = score_col_splitter(X, "Q4_Score")
        
        teams_list = list(X['Home_Team'].unique())
        team_df_list = []
        for team in teams_list:
            X_team = create_team_rolling_averages(X, team)
            team_df_list.append(X_team)
            
        X_team_data = pd.concat(team_df_list).sort_values(by = "Date")
        
        X_home = rename_team_stats(X_team_data, "Home")
        X = pd.merge(X, X_home, how = 'left', on = 'Match_ID')
        X_away = rename_team_stats(X_team_data, "Away")
        X = pd.merge(X, X_away, how = 'left', on = 'Match_ID')

        known_cols_list = ['Home_Team', 'Away_Team', 'Venue', 'Round_ID', 'Year', 'City', 'Temperature', 'Weather_Type', 'random5']
        rolling_cols_list = [x for x in list(X) if "avg" in x]
        modelling_cols = known_cols_list + rolling_cols_list

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
    