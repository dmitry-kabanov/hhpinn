"""Transformer classes."""
from sklearn.preprocessing import StandardScaler


class Transformer:
    def __init__(self, preprocessing):
        self.preprocessing = preprocessing
        
    def fit_transform(self, x, y):
        if self.preprocessing == "identity":
            xs = x
            ys = y
        elif self.preprocessing == "standardization":
            self.transformer_x = StandardScaler()
            self.transformer_x.fit(x)
            xs = self.transformer_x.transform(x)
            ys = y
        elif self.preprocessing == "standardization-both":
            raise ValueError("Incomplete; inverse transform is needed")
            self.transformer = StandardScaler()
            self.transformer.fit(x)
            xs = self.transformer.transform(x)
            self.transformer_output = StandardScaler()
            self.transformer_output.fit(y)
            ys = self.transformer_output.transform(y)
        else:
            raise ValueError("Unknown values for preprocessing")
        
        return xs, ys
    
    def transform(self, x):
        if self.preprocessing == "identity":
            return x

        return self.transformer_x.transform(x)