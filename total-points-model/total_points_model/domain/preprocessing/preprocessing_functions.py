import numpy as np
import pandas as pd

def score_col_splitter(X, score_col):
    """ Given input dataframe with a character based AFL score_col.
        Create new columns with Home, Away and Total variables for modelling.

    Args:
        X (Dataframe): Match level dataframe
        score_col (Str): Column with AFL score

    Returns:
        Dataframe: Original dataframe with score_col split into Home, Away, Total cols.
    """

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
    """ For a given dataframe, column and group, calculate the rolling average for that group.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        group (Str): Column to group rolling averages by.
        rolling_window (Int): How many games to include in the rolling average

    Returns:
        Series: Rolling average column.
    """
    
    return X.groupby(group)[column].apply(lambda x: x.shift().rolling(rolling_window).mean())

def create_rolling_average(X, column, rolling_window):
    """ For a given dataframe and column, calculate the rolling average.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        rolling_window (Int): How many games to include in the rolling average

    Returns:
        Series: Rolling average column.
    """
    
    return X[column].rolling(rolling_window).mean().shift()

def create_weighted_group_rolling_average(X, column, group, rolling_window, weights):
    """ For a given dataframe, column and group, calculate the weighted rolling average for that group.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        group (Str): Column to group rolling averages by.
        rolling_window (Int): How many games to include in the rolling average
        weights (List): Weights to apply to weighted average.

    Returns:
        Series: Rolling average column.
    """
    
    assert rolling_window == len(weights)
    
    return X.groupby(group)[column].apply(lambda x: x.shift().rolling(rolling_window).apply(lambda x: np.sum(weights*x)))

def create_weighted_rolling_average(X, column, rolling_window, weights):
    """ For a given dataframe, column, calculate the weighted rolling average.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        rolling_window (Int): How many games to include in the rolling average
        weights (List): Weights to apply to weighted average.

    Returns:
        Series: Rolling average column.
    """    
    
    assert rolling_window == len(weights)
    
    return X[column].rolling(rolling_window).apply(lambda x: np.sum(weights*x)).shift()
    
def create_exp_weighted_group_rolling_average(X, column, group, span_window):
    """ For a given dataframe, column and group, calculate the exponential weighted rolling average 
        for that group.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        group (Str): Column to group rolling averages by.
        span_window (Int): How many games to include in the rolling average

    Returns:
        Series: Rolling average column.
    """
    
    return X.groupby(group)[column].apply(lambda x: x.shift().ewm(span = span_window).mean())

def create_exp_weighted_rolling_average(X, column, span_window):
    """ For a given dataframe, column, calculate the exponential weighted rolling average.

    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        column (Str): Column to source the rolling average from.
        span_window (Int): How many games to include in the rolling average

    Returns:
        Series: Rolling average column.
    """    
    
    return X[column].ewm(span = span_window).mean().shift()

def get_team_rolling_averages(X, team, stat, rolling_window, weights=None):
    """ Converts match level home and away features into team level for and against
        features.
    Args:
        X (DataFrame): Match level dataframe with rolling average columns.
        team (Str): Home or Away team.
        stat (Str): Column to source the rolling average from.
        rolling_window (Int): How many games to include in the rolling average
        weights (List): (Optional) Weights to apply to weighted average.

    Returns:
        DataFrame : Team specific data with rolling averages calculated.
    """
    
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
    """ Given Home/Away dataframe and rolling average columns.
        Rename the Home/Away columns to For/Against for the team.

    Args:
        X (Dataframe): Team rolling average dataframe.
        home_away (Str): "Home" or "Away"

    Returns:
        X: Team rolling average dataframe with Home/Away renamed to For/Against.
    """
    
    rolling_avg_cols = [x for x in list(X) if "avg" in x]

    X_rolling = X[X['Home_Away'] == home_away]
    X_rolling_avg_cols = [x.replace("Team", home_away) for x in rolling_avg_cols]
    rename_dict = dict(zip(rolling_avg_cols, X_rolling_avg_cols))
    X_rolling = X_rolling.rename(columns=rename_dict)
    
    X_rolling = X_rolling[['Match_ID'] + X_rolling_avg_cols]
        
    return X_rolling

def merge_rolling_data(X, X_home, X_away):
    """ Merge rolling average data for home and away back onto original match data.

    Args:
        X (Dataframe): Original match data.
        X_home (Dataframe): Match data for Home team with rolling averages
        X_away (Dataframe): Match data for Away team with rolling averages

    Returns:
        Dataframe: _description_
    """
    
    X = pd.merge(X, X_home, how = 'left', on = 'Match_ID')
    X = pd.merge(X, X_away, how = 'left', on = 'Match_ID')
   
    return X

