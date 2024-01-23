import pandas as pd
import numpy as np

# Test Mappings
def test_mappings():

    from total_points_model.domain.contracts.mappings import Mappings

    mappings = Mappings

    X = pd.DataFrame({
        'Match_ID':['200501_BrisbaneLions_StKilda', '200501_NorthMelbourne_Carlton'],
        'Venue':['Gabba', 'Docklands']
    })
    X['Roof'] = X['Venue'].replace(mappings.new_feature_mappings[('Venue', 'Roof')])

    assert set(X['Roof']) == {'No Roof', 'Roof'}

# Test score_col_splitter
def test_score_col_splitter():
    
    from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter

    X = pd.DataFrame({
        'Match_ID':['200501_BrisbaneLions_StKilda', '200501_NorthMelbourne_Carlton'],
        'Q4_Score':['18.8.116 - 13.15.93', '16.9.105 - 12.13.85']
    })
    X = score_col_splitter(X, "Q4_Score")

    assert {
        'Home_Q4_Score',
        'Home_Q4_Goals',
        'Home_Q4_Behinds',
        'Home_Q4_Shots',
        'Home_Q4_Conversion',
        'Away_Q4_Score',
        'Away_Q4_Goals',
        'Away_Q4_Behinds',
        'Away_Q4_Shots',
        'Away_Q4_Conversion',
        'Total_Q4_Score',
        'Total_Q4_Goals',
        'Total_Q4_Behinds',
        'Total_Q4_Shots',
        'Total_Q4_Conversion',
    }.issubset(X.columns)

# Test create rolling average
def test_rolling_average():
    
    from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, create_rolling_average

    rolling_window = 2
    stat = "Q4_Shots"
    X = pd.DataFrame({
        'Match_ID':['200501_BrisbaneLions_StKilda', '200501_NorthMelbourne_Carlton', '200501_Melbourne_Essendon'],
        "Home_Team": ['Brisbane Lions', 'North Melbourne', 'Melbourne'],
        "Away_Team":['St Kilda', 'Carlton', 'Essendon'],
        'Q4_Score':['18.8.116 - 13.15.93', '16.9.105 - 12.13.85', '15.13.103 - 8.9.57']
    })
    X = score_col_splitter(X, "Q4_Score")

    X[f'Total_{stat}_avg{rolling_window}'] = create_rolling_average(
        X, f'Total_{stat}', rolling_window
    )

    assert X.loc[0:1, 'Total_Q4_Shots'].mean() == X.loc[2, 'Total_Q4_Shots_avg2']

# Test weighted rolling average
def test_weighted_rolling_average():
    
    from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, create_weighted_rolling_average

    rolling_window = 2
    stat = "Q4_Shots"
    weights = np.array([0.25, 0.75])
    X = pd.DataFrame({
        'Match_ID':['200501_BrisbaneLions_StKilda', '200501_NorthMelbourne_Carlton', '200501_Melbourne_Essendon'],
        "Home_Team": ['Brisbane Lions', 'North Melbourne', 'Melbourne'],
        "Away_Team":['St Kilda', 'Carlton', 'Essendon'],
        'Q4_Score':['18.8.116 - 13.15.93', '16.9.105 - 12.13.85', '15.13.103 - 8.9.57']
    })
    X = score_col_splitter(X, "Q4_Score")

    X[f'Total_{stat}_wavg{rolling_window}'] = create_weighted_rolling_average(
        X, f'Total_{stat}', rolling_window, weights
    )
    print(X[['Total_Q4_Shots', 'Total_Q4_Shots_wavg2']].head())
    assert (X.loc[0, 'Total_Q4_Shots']*weights[0] + X.loc[1, 'Total_Q4_Shots']*weights[1]) == X.loc[2, 'Total_Q4_Shots_wavg2']

# Test exp_weighted_rolling average
def test_exp_weighted_rolling_average():
    
    from total_points_model.domain.preprocessing.preprocessing_functions import score_col_splitter, create_exp_weighted_rolling_average

    span_window = 2
    alpha = 2/(span_window+1)
    stat = "Q4_Shots"
    X = pd.DataFrame({
        'Match_ID':['200501_BrisbaneLions_StKilda', '200501_NorthMelbourne_Carlton', '200501_Melbourne_Essendon'],
        "Home_Team": ['Brisbane Lions', 'North Melbourne', 'Melbourne'],
        "Away_Team":['St Kilda', 'Carlton', 'Essendon'],
        'Q4_Score':['18.8.116 - 13.15.93', '16.9.105 - 12.13.85', '15.13.103 - 8.9.57']
    })
    X = score_col_splitter(X, "Q4_Score")

    X[
        f'Total_{stat}_exp_wavg{span_window}'
    ] = create_exp_weighted_rolling_average(X, f'Total_{stat}', span_window)

    assert (X.loc[0, 'Total_Q4_Shots']*(1-alpha) + X.loc[1, 'Total_Q4_Shots']) / (1 + (1 - alpha)) == X.loc[2, 'Total_Q4_Shots_exp_wavg2']

# Test Preprocessor
def test_preprocessor():
    
    from total_points_model.domain.contracts.mappings import Mappings
    from total_points_model.domain.contracts.rolling_columns import RollingColumns
    from total_points_model.domain.preprocessing.data_preprocessor import DataPreprocessor
    
    X = pd.DataFrame({
        'Venue': {0: 'Gabba', 1: 'Docklands', 2: 'M.C.G.'},
        'Round_ID': {0: '200501', 1: '200501', 2: '200501'},
        'Match_ID': {0: '200501_BrisbaneLions_StKilda', 1: '200501_NorthMelbourne_Carlton', 2: '200501_Melbourne_Essendon'},
        'Year': {0: 2005.0, 1: 2005.0, 2: 2005.0},
        'Home_Team': {0: 'Brisbane Lions', 1: 'North Melbourne', 2: 'Melbourne'},
        'Away_Team': {0: 'St Kilda', 1: 'Carlton', 2: 'Essendon'},
        'Q4_Score': {0: '18.8.116 - 13.15.93',1: '16.9.105 - 12.13.85',2: '15.13.103 - 8.9.57'},
        'Margin': {0: 23.0, 1: 20.0, 2: 46.0},
        'Total_Game_Score': {0: 209.0, 1: 190.0, 2: 160.0},
        'Home_Win': {0: 1.0, 1: 1.0, 2: 1.0},
        'City': {0: 'Brisbane', 1: 'Melbourne', 2: 'Melbourne'},
        'Date': {0: '2005-03-24', 1: '2005-03-26', 2: '2005-03-26'},
        'Attendance': {0: 33369.0, 1: 40345.0, 2: 47849.0},
        'Temperature': {0: 18.0, 1: 18.0, 2: 18.0},
        'Weather_Type': {0: 'MOSTLY_SUNNY', 1: 'MOSTLY_SUNNY', 2: 'MOSTLY_SUNNY'},
        'Match_Status': {0: 'CONCLUDED', 1: 'CONCLUDED', 2: 'CONCLUDED'},
        'Weather_Description': {0: 'Mostly Sunny',1: 'Mostly Sunny',2: 'Mostly Sunny'},
        'Ground_Width': {0: 138, 1: 129, 2: 141},
        'Ground_Length': {0: 156, 1: 160, 2: 160},
        'Home_Ground': {0: 'Primary Home', 1: 'Primary Home', 2: 'Primary Home'}
    })
    
    preprocessor = DataPreprocessor(Mappings=Mappings, rolling_dict=RollingColumns.rolling_dict)
    
    preprocessor.fit(X)
    X_preproc = preprocessor.transform(X)

    expected_columns = [
        'Match_ID',
        'Year',
        'Round',
        'Finals',
        'Primary_Home',
        'Home_Team_Within_State',
        'Away_Team_Within_State',
        'ELO_Home',
        'ELO_Away',
        'ELO_diff',
        'ELO_abs_diff',
        'ELO_probs_Home',
        'ELO_probs_Away',
        'ELO_probs_diff',
        'ELO_probs_abs_diff',
        'Home_Total_Q4_Score_exp_wavg10',
        'Away_Total_Q4_Score_exp_wavg10',
        'Home_Total_Q4_Shots_exp_wavg10',
        'Away_Total_Q4_Shots_exp_wavg10',
        'Kicking_Weather_Good Kicking',
        'State_Queensland',
        'State_Victoria',
        'Victoria_Not Victoria',
        'Victoria_Victoria',
        'Roof_No Roof',
        'Roof_Roof'
    ]
    print(set(X_preproc.columns))
    print(set(expected_columns))
    
    assert set(X_preproc.columns) == set(expected_columns)


