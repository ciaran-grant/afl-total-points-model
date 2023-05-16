from total_points_model.domain.contracts.mappings import Mappings

class DataMapper:
    def __init__(self, mappings):
        self.Mappings = Mappings()
        self.mapping_dict = self.Mappings.mappings
        
    def transform(self, X):
        
        X = X.replace(self.mapping_dict)
        
        return X