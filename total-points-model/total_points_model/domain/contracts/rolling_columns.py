from dataclasses import dataclass
import numpy as np

@dataclass
class RollingColumns:
    """ Holds columns and information for creating rolling averages in tuples.
        "name" : ("column_name", rolling_window, weights)
    """
    rolling_dict = {
        "score10" : ("Q4_Score", 10, np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.25])),
        "shots10" : ("Q4_Shots", 10, np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.15, 0.25])),
    }
