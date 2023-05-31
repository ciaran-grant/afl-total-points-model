from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, get_team_rolling_averages, rename_rolling_columns, merge_rolling_data
from total_points_model.domain.preprocessing.elo_calculator import calculate_elo_ratings, convert_elo_dict_to_dataframe, merge_elo_ratings
from total_points_model.domain.contracts.modelling_data_contract import ModellingDataContract
from sklearn.base import BaseEstimator, TransformerMixin

import pandas as pd
import numpy as np

from total_points_model.domain.contracts.mappings import Mappings

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """ Preprocessing class and functions for training total game score model.
    """
    
    def __init__(self, Mappings, rolling_dict):
        """ Specify mappings and rolling average columns to create.

        Args:
            Mappings (Mappings): Mappings object specifying mapping and transformations.
            rolling_dict (Dict): Dictionary specifying columns and types of rolling average columns.
        """
        self.Mappings = Mappings
        self.rolling_dict = rolling_dict
        self.ModellingDataContract = ModellingDataContract
        
    def create_team_rolling_average_features(self, X, column, rolling_window, weights):
        """ Converts match level home and away features into team level for and against
            features. Then calculates rolling average as specified and merges back onto
            match level data.
        Args:
            X (DataFrame): Match level dataframe with rolling average columns.
            column (Str): Column to source the rolling average from.
            rolling_window (Int): How many games to include in the rolling average
            weights (List): (Optional) Weights to apply to weighted average.

        Returns:
            DataFrame : Input data with rolling average columns merged on.
        """
    
        team_data_list = []
        for team in self.ModellingDataContract.team_list:
            team_data = get_team_rolling_averages(X, team, column, rolling_window, weights)
            team_data_list.append(team_data)
        rolling_data = pd.concat(team_data_list, axis=0).sort_index()

        home_rolling_data = rename_rolling_columns(rolling_data, "Home")
        away_rolling_data = rename_rolling_columns(rolling_data, "Away")

        X = merge_rolling_data(X, home_rolling_data, away_rolling_data)
        
        return X
    
    def create_elo_rating_factor(self, X):
        """ Given the input data, calculates a basic ELO rating for both Home and Away teams.
            Creates ELO_home, ELO_away, ELO_probs_home, ELO_probs_away.
            Merges back onto input data.

        Args:
            X (DataFrame): Dataframe including match scores and teams to calculate ELOs from.

        Returns:
            DataFrame: Input dataframe returned with ELO columns merged on.
        """
        
        elos, elo_dict, elo_probs  = calculate_elo_ratings(X, k_factor=ModellingDataContract.ELO_K_FACTOR)
        
        X = merge_elo_ratings(X, elos, elo_probs)
        
        return X
        
    def fit(self, X):
        """ Fits preprocessor to training data.
            Learns expected columns and mean imputations. 

        Args:
            X (Dataframe): Training dataframe to fit preprocessor to.

        Returns:
            self: Preprocessor learns expected colunms and means to impute.
        """
        
        X_copy = X.copy()
        
        # Feature Engineering
        X_copy['Date'] = pd.to_datetime(X_copy['Date'])
        X_copy['Round'] = X_copy['Round_ID'].apply(lambda x: x[-2:])
        X_copy['Finals'] = np.where(X_copy['Round_ID'].str.contains('F'), True, False)
        X_copy['Primary_Home'] = np.where(X_copy['Home_Ground'] == "Primary Home", True, False)
        
        # Feature Grouping and Mappings
        X_copy['Kicking_Weather'] = X_copy['Weather_Description'].replace(self.Mappings.new_feature_mappings[("Weather_Description", "Kicking_Weather")])
        X_copy['State'] = X_copy['City'].replace(self.Mappings.new_feature_mappings[('City', 'State')])
        X_copy['Victoria'] = X_copy['City'].replace(self.Mappings.new_feature_mappings[('City', 'Victoria')])
        X_copy['Roof'] = X_copy['Venue'].replace(self.Mappings.new_feature_mappings[('Venue', 'Roof')])

        X_copy['Home_Team_State'] = X_copy['Home_Team'].replace(self.Mappings.meta_mappings[('Team', 'State')])
        X_copy['Away_Team_State'] = X_copy['Away_Team'].replace(self.Mappings.meta_mappings[('Team', 'State')])
        X_copy['Home_Team_Within_State'] = np.where(X_copy['State'] == X_copy['Home_Team_State'], True, False)
        X_copy['Away_Team_Within_State'] = np.where(X_copy['State'] == X_copy['Away_Team_State'], True, False)

        # Apply self mappings
        X_copy = X_copy.replace(self.Mappings.transformation_mappings)

        # Scoring Columns
        X_copy = score_col_splitter(X_copy, "Q4_Score")
        # Rolling Averages by Team
        for name, stat in self.rolling_dict.items():
            X_copy = self.create_team_rolling_average_features(X_copy, column=stat[0], rolling_window=stat[1], weights=stat[2])

        # ELO Ratings
        # print(list(X_copy))
        X_copy = self.create_elo_rating_factor(X_copy)
        # print(list(X_copy))

        self.rolling_cols_list = [x for x in list(X_copy) if "avg" in x]
        self.modelling_cols = ModellingDataContract.modelling_feature_list + self.rolling_cols_list

        # Keep only modelling columns and ID
        X_copy = X_copy[self.modelling_cols]
        
        # Fitting to training data
        self.train_set_means = X_copy.mean()
        self.expected_dummy_cols = list(pd.get_dummies(X_copy))
                        
        return self
    
    def transform(self, X):
        """ Applies transformations and preprocessing steps to dataframe.

        Args:
            X (Dataframe): Training or unseen data to transform.

        Returns:
            Dataframe: Transformed data with modelling columns and no missing values.
        """
        
        # Feature Engineering
        X['Date'] = pd.to_datetime(X['Date'])
        X['Round'] = X['Round_ID'].apply(lambda x: x[-2:])
        X['Finals'] = np.where(X['Round_ID'].str.contains('F'), True, False)
        X['Primary_Home'] = np.where(X['Home_Ground'] == "Primary Home", True, False)
        
        # Feature Grouping and Mappings
        X['Kicking_Weather'] = X['Weather_Description'].replace(self.Mappings.new_feature_mappings[("Weather_Description", "Kicking_Weather")])
        X['State'] = X['City'].replace(self.Mappings.new_feature_mappings[('City', 'State')])
        X['Victoria'] = X['City'].replace(self.Mappings.new_feature_mappings[('City', 'Victoria')])
        X['Roof'] = X['Venue'].replace(self.Mappings.new_feature_mappings[('Venue', 'Roof')])

        X['Home_Team_State'] = X['Home_Team'].replace(self.Mappings.meta_mappings[('Team', 'State')])
        X['Away_Team_State'] = X['Away_Team'].replace(self.Mappings.meta_mappings[('Team', 'State')])
        X['Home_Team_Within_State'] = np.where(X['State'] == X['Home_Team_State'], True, False)
        X['Away_Team_Within_State'] = np.where(X['State'] == X['Away_Team_State'], True, False)

        # Apply self mappings
        X = X.replace(self.Mappings.transformation_mappings)

        # Scoring Columns
        X = score_col_splitter(X, "Q4_Score")
        # Rolling Averages by Team
        for name, stat in self.rolling_dict.items():
            X = self.create_team_rolling_average_features(X, column=stat[0], rolling_window=stat[1], weights=stat[2])

        # ELO Ratings
        X = self.create_elo_rating_factor(X)
        
        self.rolling_cols_list = [x for x in list(X) if "avg" in x]
        self.modelling_cols = ModellingDataContract.modelling_feature_list + self.rolling_cols_list

        # Keep only modelling columns and ID
        X = X[[ModellingDataContract.ID_COL] + self.modelling_cols]
        
        # Applying transformations
        X = X.fillna(self.train_set_means)        
        X_dummies = pd.get_dummies(X[self.modelling_cols])
        for col in list(self.expected_dummy_cols):
            if col not in list(X_dummies):
                X_dummies[col] = 0
                
        X_dummies = X_dummies[self.expected_dummy_cols]

        X = pd.concat([X[ModellingDataContract.ID_COL], X_dummies], axis = 1)

        return X
    