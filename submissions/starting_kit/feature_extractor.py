import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline


class CountOrdinalEncoder(OrdinalEncoder):
    """Encode categorical features as an integer array
    usint count information.
    """

    def __init__(self, categories='auto', dtype=np.float64):
        self.categories = categories
        self.dtype = dtype

    def fit(self, X, y=None):
        """Fit the OrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self
        """
        super().fit(X)
        X_list = self._check_X(X)
        # now we'll reorder by counts
        for k, cat in enumerate(self.categories_):
            counts = []
            for c in cat:
                counts.append(np.sum(X_list[k] == c))
            order = np.argsort(counts)
            self.categories_[k] = cat[order]
        return self


class FeatureExtractor(object):
    """
    string columns:
    'Name', 'Club','Nationality'

    obj columns:
    'Value','Wage', 'Preferred Foot',  'Body Type', 'Position',  'Joined',
    'Loaned From', 'Contract Valid Until', 'Height', 'Weight', 'LS', 'ST',
    'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM',
    'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB',
    'RCB', 'RB',  'Release Clause'

    numerical columns(float64):
    'International Reputation', 'Weak Foot', 'Skill Moves', 'Jersey Number',
    'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
    'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',
    'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
    'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
    'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
    'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
    'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'

    numerical columns(int64):
    'ID',  'Age',

    """

    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        num_cols = ['Age', 'International Reputation', 'Weak Foot', 'Skill Moves',
                           'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing',
                           'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing',
                           'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions',
                           'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
                           'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
                           'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',
                           'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

        cat_cols = ['Preferred Foot',  'Body Type', 'Position']

        drop_cols = ['Name', 'Club', 'Nationality']

        X_selected = X_df[num_cols+cat_cols+drop_cols]

        numeric_transformer = Pipeline(steps=[
            ('impute', SimpleImputer(strategy='median'))])

        cat_encoder = CountOrdinalEncoder()

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', make_pipeline(SimpleImputer(strategy='constant', fill_value='nan'), cat_encoder), cat_cols),
                ('num', numeric_transformer, num_cols),
                ('drop cols', 'drop', drop_cols+cat_cols)
            ])

        X_array = preprocessor.fit_transform(X_selected)
        return X_array
