from dataclasses import dataclass

@dataclass
class OptunaXGBParamGrid:
    trials: int = 1000
    verbosity: int = 1
    error: str = "reg:squarederror"
    eta_min: float = 0.0001
    eta_max: float = 1.0
    max_depth_min: int = 2
    max_depth_max: int = 20
    max_depth_step: int = 2
    min_child_weight_min: int = 2
    min_child_weight_max: int = 20
    min_child_weight_step: int = 5
    gamma_min: float = 0.0001
    gamma_max: float = 1
    lambda_min: float = 0.0001
    lambda_max: float = 1
    alpha_min: float = 0.0001
    alpha_max: float = 1
    subsample_min: float = 0.2
    subsample_max: float = 0.9
    subsample_step: float = 0.05
    colsample_bytree_min: float = 0.2
    colsample_bytree_max: float = 0.9
    colsample_bytree_step: float = 0.5
    validation_size: float = 0.2
      