from dataclasses import dataclass

@dataclass
class ModellingDataContract:
    
    ID_COL = "Match_ID"
    RESPONSE = "Total_Game_Score"
    TRAIN_TEST_SPLIT_COL = "ModellingFilter2019"
    
    ELO_K_FACTOR = 30
    
    team_list = [
        'Adelaide',
        'Brisbane Lions',
        'Carlton',
        'Collingwood',
        'Essendon',
        'Fremantle',
        'Geelong',
        'Gold Coast',
        'Greater Western Sydney',
        'Hawthorn',
        'Melbourne',
        'North Melbourne',
        'Port Adelaide',
        'Richmond',
        'St Kilda',
        'Sydney',
        'West Coast',
        'Western Bulldogs'
    ]
    
    raw_modelling_cols = [
        "Round_ID",
        "Home_Ground",
        "City",
        "Venue",
        "Home_Team",
        "Away_Team",
        "Q4_Score",
        "Weather_Description"
        ]
    
    modelling_feature_list = [
        "Year",
        "Round",
        'Kicking_Weather',
        'State',
        'Victoria',
        'Roof',
        'Finals',
        'Primary_Home',
        'Home_Team_Within_State',
        'Away_Team_Within_State',
        "ELO_Home",
        "ELO_Away",
        'ELO_diff',
        'ELO_abs_diff',
        'ELO_probs_Home',
        'ELO_probs_Away',
        'ELO_probs_diff',
        'ELO_probs_abs_diff'
    ]