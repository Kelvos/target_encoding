from sklearn.base import BaseEstimator, TransformerMixin


class KappaEncoder(BaseEstimator, TransformerMixin):
    def __init__(self,kappa = 0):
        self.kappa = kappa
        
    def encode(self, X):
        
        
        for col in self.columns:
            unique_vals = np.unique(X[:,col])
            X_train_col_np = self.train_col_values[col]
            y_train_np = self.train_target_values
            
            
            for val in unique_vals:
                distances = np.abs(val-X_train_col_np)
                weights = 1/(1+distances)**self.kappa
                imputed_value = np.sum(y_train_np*weights)/np.sum(weights)
                self.missing_values[(col, val)] = imputed_value
                
            for val in set(np.unique(X_train_col_np)) - set(unique_vals):
                distances = np.abs(val-X_train_col_np)
                weights = 1/(1+distances)**self.kappa
                imputed_value = np.sum(y_train_np*weights)/np.sum(weights)
                self.missing_values[(col, val)] = imputed_value
        
        for (col,val), imputed_value in self.missing_values.items():
            X[X[:,col] == val,col] = imputed_value
        return X
    
    def fit(self,X,y):
        self.columns = list(range(0,X.shape[1]))
        self.missing_values = {}
        self.train_col_values = {col:X[:,col] for col in self.columns}
        self.train_target_values = y
        
    def transform(self, X , y = None):
        return self.encode(X.copy())
    
    def fit_transform(self, X , y):
        self.fit(X, y)
        return self.transform(X)
#sklearn compatible target encoder
