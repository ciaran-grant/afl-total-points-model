import numpy as np
import pandas as pd

def score_col_splitter(X, score_col):

    quarter = score_col.split("_")[0]
    
    X['Home_'+score_col] = X[score_col].apply(lambda x: x.split(" - ")[0].split(".")[-1]).astype(int)
    X['Home_'+quarter+'_Goals'] = X[score_col].apply(lambda x: x.split(" - ")[0].split(".")[0]).astype(int)
    X['Home_'+quarter+'_Behinds'] = X[score_col].apply(lambda x: x.split(" - ")[0].split(".")[1]).astype(int)
    X['Home_'+quarter+'_Shots'] = X['Home_'+quarter+'_Goals'] + X['Home_'+quarter+'_Behinds']
    X['Home_'+quarter+'_Conversion'] = X['Home_'+quarter+'_Goals'] / X['Home_'+quarter+'_Shots']
    
    X['Away_'+score_col] = X[score_col].apply(lambda x: x.split(" - ")[1].split(".")[-1]).astype(int)
    X['Away_'+quarter+'_Goals'] = X[score_col].apply(lambda x: x.split(" - ")[1].split(".")[0]).astype(int)
    X['Away_'+quarter+'_Behinds'] = X[score_col].apply(lambda x: x.split(" - ")[1].split(".")[1]).astype(int)
    X['Away_'+quarter+'_Shots'] = X['Away_'+quarter+'_Goals'] + X['Away_'+quarter+'_Behinds']
    X['Away_'+quarter+'_Conversion'] = X['Away_'+quarter+'_Goals'] / X['Away_'+quarter+'_Shots']
    
    X['Total_'+score_col] = X['Home_'+score_col] + X['Away_'+score_col]
    X['Total_'+quarter+'_Goals'] = X['Home_'+quarter+'_Goals'] + X['Away_'+quarter+'_Goals']
    X['Total_'+quarter+'_Behinds'] = X['Home_'+quarter+'_Behinds'] + X['Away_'+quarter+'_Behinds']
    X['Total_'+quarter+'_Shots'] = X['Home_'+quarter+'_Shots'] + X['Away_'+quarter+'_Shots']
    X['Total_'+quarter+'_Conversion'] = X['Total_'+quarter+'_Goals'] / X['Total_'+quarter+'_Shots']
        
    return X

def create_group_rolling_average(X, column, group, rolling_window):
    
    return X.groupby(group)[column].apply(lambda x: x.shift().rolling(rolling_window).mean())

def create_rolling_average(X, column, rolling_window):
    
    return X[column].rolling(rolling_window).mean().shift()

def create_weighted_group_rolling_average(X, column, group, rolling_window, weights):
    
    assert rolling_window == len(weights)
    
    return X.groupby(group)[column].apply(lambda x: x.shift().rolling(rolling_window).apply(lambda x: np.sum(weights*x)))

def create_weighted_rolling_average(X, column, rolling_window, weights):
    
    assert rolling_window == len(weights)
    
    return X[column].rolling(rolling_window).apply(lambda x: np.sum(weights*x)).shift()
    
def create_exp_weighted_group_rolling_average(X, column, group, span_window):
    
    return X.groupby(group)[column].apply(lambda x: x.shift().ewm(span = span_window).mean())

def create_exp_weighted_rolling_average(X, column, span_window):
    
    return X[column].ewm(span = span_window).mean().shift()

def get_team_rolling_averages(X, team, stat, rolling_window, weights):
    
    stat_cols = ["Total_"+stat, "Home_"+stat, "Away_"+stat]
    
    X_team = X[(X['Home_Team'] == team )| (X['Away_Team'] == team)]
    X_team = X_team[["Match_ID", "Home_Team", "Away_Team"] + stat_cols]
    X_team['Team'] = team
    X_team['Home_Away'] = np.where(X_team['Home_Team'] == team, 'Home', 'Away')
    
    X_team['Team_Total_'+stat] = X_team['Total_'+stat]
    X_team['Team_For_'+stat] = np.where(X_team['Home_Team'] == team, X_team['Home_'+stat], X_team['Away_'+stat])
    X_team['Team_Against_'+stat] = np.where(X_team['Home_Team'] == team, X_team['Away_'+stat], X_team['Home_'+stat])
    
    # X_team['Team_Total_'+stat+"_avg"+str(rolling_window)] = create_rolling_average(X_team, 'Team_Total_'+stat, rolling_window)
    # X_team['Team_For_'+stat+"_avg"+str(rolling_window)] = create_rolling_average(X_team, 'Team_For_'+stat, rolling_window)
    # X_team['Team_Against_'+stat+"_avg"+str(rolling_window)] = create_rolling_average(X_team, 'Team_Against_'+stat, rolling_window)

    # X_team['Team_Total_'+stat+"_wavg"+str(rolling_window)] = create_weighted_rolling_average(X_team, 'Team_Total_'+stat, rolling_window, weights)
    # X_team['Team_For_'+stat+"_wavg"+str(rolling_window)] = create_weighted_rolling_average(X_team, 'Team_For_'+stat, rolling_window, weights)
    # X_team['Team_Against_'+stat+"_wavg"+str(rolling_window)] = create_weighted_rolling_average(X_team, 'Team_Against_'+stat, rolling_window, weights)

    X_team['Team_Total_'+stat+"_exp_wavg"+str(rolling_window)] = create_exp_weighted_rolling_average(X_team, 'Team_Total_'+stat, rolling_window)
    # X_team['Team_For_'+stat+"_exp_wavg"+str(rolling_window)] = create_exp_weighted_rolling_average(X_team, 'Team_For_'+stat, rolling_window)
    # X_team['Team_Against_'+stat+"_exp_wavg"+str(rolling_window)] = create_exp_weighted_rolling_average(X_team, 'Team_Against_'+stat, rolling_window)
   
    return X_team

def rename_rolling_columns(X, home_away):
    
    rolling_avg_cols = [x for x in list(X) if "avg" in x]

    X_rolling = X[X['Home_Away'] == home_away]
    X_rolling_avg_cols = [x.replace("Team", home_away) for x in rolling_avg_cols]
    rename_dict = dict(zip(rolling_avg_cols, X_rolling_avg_cols))
    X_rolling = X_rolling.rename(columns=rename_dict)
    
    X_rolling = X_rolling[['Match_ID'] + X_rolling_avg_cols]
        
    return X_rolling

def merge_rolling_data(X, X_home, X_away):
    
    X = pd.merge(X, X_home, how = 'left', on = 'Match_ID')
    X = pd.merge(X, X_away, how = 'left', on = 'Match_ID')
   
    return X

