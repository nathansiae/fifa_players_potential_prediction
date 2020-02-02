import os
import rampwf as rw
import pandas as pd
from rampwf.workflows import FeatureExtractorRegressor
from rampwf.score_types.base import BaseScoreType
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error

problem_title = 'fifa_players_potential_prediction'
_target_column_name = 'Potential'

Predictions = rw.prediction_types.make_regression()


class FIFA(FeatureExtractorRegressor):
    def __init__(self, workflow_element_names=['feature_extractor', 'regressor']):
        super(FIFA, self).__init__(workflow_element_names)
        self.element_names = workflow_element_names


class MaeError(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='mean aboslute error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        if isinstance(y_true, pd.Series):
            y_true = y_true.values

        loss = mean_absolute_error(y_true, y_pred)
        return loss


workflow = FIFA()

score_types = [MaeError()]


def get_cv(X, y):
    cv = GroupShuffleSplit(n_splits=6, test_size=0.30, random_state=42)
    return cv.split(X, y, groups=X['Age'])


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), low_memory=False, index_col=0)
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    return X_df, y_array


def get_train_data(path='.'):
    f_name = 'fifa_train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'fifa_test.csv'
    return _read_data(path, f_name)
