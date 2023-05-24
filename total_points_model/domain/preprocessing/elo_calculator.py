from total_points_model.domain.contracts.modelling_data_contract import ModellingDataContract
import pandas as pd

def calculate_elo_ratings(data, k_factor):
    
    # Initialise a dictionary with default elos for each team
    elo_dict = {team: 1500 for team in ModellingDataContract.team_list}
    elos, elo_probs = {}, {}
    
    for index, row in data.iterrows():
        game_id = row['Match_ID']
        margin = row['Margin']
        
        if game_id in elos.keys():
            continue
        
        home_team = row['Home_Team']
        away_team = row['Away_Team']
        
        home_team_elo = elo_dict[home_team]
        away_team_elo = elo_dict[away_team]
        
        prob_win_home = 1 / (1 + 10**((away_team_elo - home_team_elo) / 400))
        prob_win_away = 1 - prob_win_home
        
        elos[game_id] = [home_team_elo, away_team_elo]
        elo_probs[game_id] = [prob_win_home, prob_win_away]
        
        if margin > 0:
            new_home_team_elo = home_team_elo + k_factor*(1 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0 - prob_win_away)
        elif margin < 0:
            new_home_team_elo = home_team_elo + k_factor*(0 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(1 - prob_win_away)
        elif margin == 0:
            new_home_team_elo = home_team_elo + k_factor*(0.5 - prob_win_home)
            new_away_team_elo = away_team_elo + k_factor*(0.5 - prob_win_away)
            
        elo_dict[home_team] = new_home_team_elo
        elo_dict[away_team] = new_away_team_elo

    
    return elos, elo_dict, elo_probs

def convert_elo_dict_to_dataframe(elos, elo_probs):
    
    elo_df = pd.DataFrame(list(elos.items()), columns = ['Match_ID', 'ELO_list'])
    elo_df[['ELO_Home', 'ELO_Away']] = elo_df['ELO_list'].tolist()
    elo_df['ELO_diff'] = elo_df['ELO_Home'] - elo_df['ELO_Away']
    elo_df['ELO_abs_diff'] = abs(elo_df['ELO_diff'])
    elo_df = elo_df.drop(columns = ['ELO_list'])
    
    elo_probs_df = pd.DataFrame(list(elo_probs.items()), columns = ['Match_ID', 'ELO_probs_list'])
    elo_probs_df[['ELO_probs_Home', 'ELO_probs_Away']] = elo_probs_df['ELO_probs_list'].tolist()
    elo_probs_df['ELO_probs_diff'] = elo_probs_df['ELO_probs_Home'] - elo_probs_df['ELO_probs_Away']
    elo_probs_df['ELO_probs_abs_diff'] = abs(elo_probs_df['ELO_probs_diff'])
    elo_probs_df = elo_probs_df.drop(columns = ['ELO_probs_list'])
    
    return elo_df, elo_probs_df

def merge_elo_ratings(X, elos, elo_probs):
    
    elo_df, elo_probs_df = convert_elo_dict_to_dataframe(elos, elo_probs)
    
    X = pd.merge(X, elo_df, how = 'left', on = 'Match_ID')
    X = pd.merge(X, elo_probs_df, how = 'left', on = 'Match_ID')
    
    return X