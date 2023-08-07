import numpy as np
import pandas as pd
import numpy as np

class Survival:

    def __init__(self):
        pass    

    def predict_survival_function(self, model, X_test, y_test, lower, upper):
#        lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        if not model.__class__.__name__ == 'DeepCoxPH':
            times = np.arange(np.ceil(lower + 1), np.floor(upper - 1), dtype=int)
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_survival_function(X_test).T
        elif model.__class__.__name__ == 'DeepCoxPH':
            x= X_test.to_numpy()
            times= lower
            surv_prob = model.predict_survival(x, times).T
            return surv_prob            
        else:
            surv_prob = np.row_stack([fn(times) for fn in model.predict_survival_function(X_test)])
        return pd.DataFrame(surv_prob, columns=times)

    def predict_hazard_function(self, model, X_test, y_test, lower, upper):
#        lower, upper = np.percentile(y_test[y_test.dtype.names[1]], [10, 90])
        times = np.arange(np.ceil(lower), np.floor(upper), dtype=int)
        if model.__class__.__name__ == 'WeibullAFTFitter':
            surv_prob = model.predict_cumulative_hazard(X_test)
        else:
            surv_prob = np.row_stack([fn(times) for fn in model.predict_cumulative_hazard_function(X_test)])
        return pd.DataFrame(surv_prob, columns=times)