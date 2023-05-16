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

def create_team_rolling_averages(X, team, rolling_window = 2):
    
    X_team = X[(X['Home_Team'] == team) | (X['Away_Team'] == team)]
    X_team['Team'] = team
    X_team['Home'] = np.where(X_team['Home_Team'] == team, "Home", 'Away')
    X_team = X_team.sort_values(by = "Date")
    
    quarters_list = ['Q4']
    stats_list = ['Score', 'Goals', 'Behinds', 'Shots', 'Conversion']
    quarter_stats_list = [x+"_"+y for x in quarters_list for y in stats_list]
    
    for stat in quarter_stats_list:
        X_team['Team_Total_'+stat] = X_team['Total_'+stat]
        X_team['Team_Att_'+stat] = np.where(X_team['Home_Team'] == team, X_team['Home_'+stat], X_team['Away_'+stat])
        X_team['Team_Def_'+stat] = np.where(X_team['Home_Team'] == team, X_team['Away_'+stat], X_team['Home_'+stat])
        
    home_cols = [x for x in list(X_team) if "Home_Q" in x]
    away_cols = [x for x in list(X_team) if "Away_Q" in x]
    
    numeric_cols = list(X_team.select_dtypes('number'))
    rolling_cols = [x for x in numeric_cols if "Team" in x] #+ [x for x in numeric_cols if "Total" in x]
    
    rolling_team_data = X_team.groupby('Team')[rolling_cols].rolling(rolling_window).mean()
    rolling_team_data.columns = [x+"_avg" + str(rolling_window) for x in rolling_cols]
    rolling_team_data = rolling_team_data.reset_index(level = [0])
    rolling_team_data = rolling_team_data.drop(columns=['Team'])
    
    X_team = pd.merge(X_team, rolling_team_data, how = 'left', left_index=True, right_index=True)
    
    return X_team

def rename_team_stats(X, home = "Home", rolling_window = 2):
    
    X_team = X[X['Home'] == home]
    
    avg_cols = [x for x in list(X_team) if "avg"+str(rolling_window) in x]
    team_total_cols = [x for x in avg_cols if 'Team_Total' in x]
    team_att_cols = [x for x in avg_cols if 'Team_Att' in x]
    team_def_cols = [x for x in avg_cols if 'Team_Def' in x]
    
    total_cols = [x.replace("Team_Total", home+"_Total") for x in team_total_cols]
    att_cols = [x.replace("Team_Att", home+"_Att") for x in team_att_cols]
    def_cols = [x.replace("Team_Def", home+"_Def") for x in team_def_cols]

    team_total_dict = dict(zip(team_total_cols, total_cols))
    team_att_dict = dict(zip(team_att_cols, att_cols))
    team_def_dict = dict(zip(team_def_cols, def_cols))
    
    X_team = X_team.rename(columns=team_total_dict)
    X_team = X_team.rename(columns=team_att_dict)
    X_team = X_team.rename(columns=team_def_dict)
    
    X_team = X_team[['Match_ID']+total_cols+att_cols+def_cols]
    
    return X_team