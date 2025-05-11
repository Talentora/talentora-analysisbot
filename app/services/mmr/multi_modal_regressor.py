import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

class MultiModalRegressor:
    def __init__(self, data, target, modalities=['face', 'prosody', 'language'], 
                 svr_params=None, meta_params=None, num_top_features=41, k_fold_splits = 6):
        """
        Parameters:
        - data: dict where keys are modality names and values are feature matrices (DataFrames or numpy arrays)
        - target: array-like of target values
        - modalities: list of modality names (e.g., ['face', 'prosody', 'language'])
        - svr_params: dictionary of grid search parameters for the SVR models (default provided if None)
        - meta_params: dictionary of grid search parameters for the meta model (default provided if None)
        - num_top_features: number of top features to select based on Ridge regression coefficients
        """
        self.data = data
        self.target = target
        self.modalities = modalities
        self.num_top_features = num_top_features

        # Default grid parameters for SVR models if none are provided
        self.svr_params = svr_params if svr_params is not None else {
            'svr__C': [ 0.1, 0.2, 0.8, 0.9, 1, 10, 11, 12],
            'svr__gamma': ['scale', 'auto'],
            'svr__kernel': ['rbf']
        }
        # Default grid parameters for the meta model (using Linear Regression as an example)
        self.meta_params = meta_params if meta_params is not None else {
            'meta__fit_intercept': [True, False]
        }

        # To hold the best SVR pipelines for each modality and their parameters
        self.svr_models = {}
        self.svr_best_params = {}
        # To hold out-of-fold predictions for each modality
        self.oof_predictions = {mod: np.zeros(len(target)) for mod in modalities}
        # To hold the meta model and its best parameters
        self.meta_model = None
        self.meta_best_params = None
        self.k_fold_splits = k_fold_splits
        
        # To store the selected feature columns for each modality
        self.selected_features = {}

    def select_features_ridge(self):
        """
        Runs a Ridge regression on the training data of each modality to select the top features.
        The selected features are stored and the training data is updated to include only these columns.
        """
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            ridge = Ridge(alpha=0.19)
            ridge.fit(X, y)
            # Rank features by the absolute value of their coefficients
            coefs = np.abs(ridge.coef_)
            # Get indices for the top features
            top_indices = np.argsort(coefs)[-self.num_top_features:]
            # If X is a DataFrame, use column names; if numpy array, use indices.
            if isinstance(X, pd.DataFrame):
                top_features = list(X.columns[top_indices])
            else:
                top_features = top_indices
            self.selected_features[mod] = top_features
            # Update the modality data to only include the selected features.
            if isinstance(X, pd.DataFrame):
                self.data[mod] = X[top_features]
            else:
                self.data[mod] = X[:, top_indices]
            print(f"Selected features for {mod}: {self.selected_features[mod]}")

    def build_svr_pipeline(self):
        """Constructs an SVR pipeline for each modality using only the selected features."""
        pipelines = {}
        for mod in self.modalities:
            # Note: PCA has been removed since feature selection is done via Ridge.
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR())
            ])
            pipelines[mod] = pipeline
        self.svr_models = pipelines

    def grid_search_svr(self):
        """Performs grid search with CV for each modality's SVR pipeline."""
        best_estimators = {}
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            pipeline = self.svr_models[mod]
            kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)
            grid_search = GridSearchCV(pipeline, self.svr_params, cv=kf)
            grid_search.fit(X, y)
            best_estimators[mod] = grid_search.best_estimator_
            self.svr_best_params[mod] = grid_search.best_params_
            print(f"Best params for {mod} modality: {grid_search.best_params_}")
        self.svr_models = best_estimators

    def generate_oof_predictions(self):
        """
        Generates out-of-fold (OOF) predictions using CV for each modality.
        Each prediction is obtained from a model that did not see the sample during training.
        """
        kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)
        n_samples = len(self.target)
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            preds = np.zeros(n_samples)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train = y[train_index]  # Use numpy indexing for target array
                model = self.svr_models[mod]
                model.fit(X_train, y_train)
                preds[test_index] = model.predict(X_test)
            self.oof_predictions[mod] = preds
        return self.oof_predictions

    def train_meta_model(self):
        """
        Trains the meta model (stacked ensemble) using the OOF predictions from each modality.
        """
        # Stack OOF predictions to form meta features
        meta_features = np.column_stack([self.oof_predictions[mod] for mod in self.modalities])
        y = self.target
        # Here, we use a pipeline with Linear Regression as an example
        self.meta_model = Pipeline([
            ('scaler', StandardScaler()),
            ('meta', LinearRegression())
        ])
        self.meta_model.fit(meta_features, y)

    def grid_search_meta(self):
        """
        Performs grid search with CV for the meta model based on the combined predictions.
        """
        meta_features = np.column_stack([self.oof_predictions[mod] for mod in self.modalities])
        y = self.target
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('meta', LinearRegression())
        ])
        kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipeline, self.meta_params, cv=kf)
        grid_search.fit(meta_features, y)
        self.meta_model = grid_search.best_estimator_
        self.meta_best_params = grid_search.best_params_
        print(f"Best params for meta model: {grid_search.best_params_}")

    def predict(self, new_data):
        """
        Predicts target values for new data.
        For each modality, the new data is reduced to the selected features before prediction.
        Returns the predicted target value and individual modality predictions as a dictionary.
        """
        if isinstance(new_data, dict):
            # Handle dictionary input (original case)
            modality_preds = {}
            modality_arrays = []
            
            for mod in self.modalities:
                X_new = new_data[mod]
                # Ensure the new data has the same columns as selected during training
                if isinstance(X_new, pd.DataFrame) and mod in self.selected_features:
                    X_new = X_new[self.selected_features[mod]]
                
                # Get prediction for this modality
                pred = self.svr_models[mod].predict(X_new)
                # Store the prediction in the dictionary with the modality as key
                modality_preds[mod] = float(round(pred[0], 2))
                # Keep track of the arrays for meta_features
                modality_arrays.append(pred)
            
            # Stack the predictions for meta model
            meta_features = np.column_stack(modality_arrays)
        else:
            # Handle array input (from process_prediction)
            meta_features = new_data
            # In this case, we don't have individual modality predictions
            modality_preds = {}
            
        # Use the meta model to predict
        overall_score = float(round(self.meta_model.predict(meta_features)[0], 2))
        
        return overall_score, modality_preds

    def evaluate(self, true_values, predictions):
        """
        Computes evaluation metrics for regression: MSE, MAE, and R^2 score.
        """
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
        return mse, mae, r2
    
    def save_model(self, filename):
        """
        Save the current instance of MultiModalRegressor (including all models, weights, and hyperparameters) to a file.
        The file will be saved in the mmr folder.
        """
        # Get the absolute path of the mmr folder
        mmr_folder = Path(__file__).parent
        # Create the full path for the file
        file_path = mmr_folder / filename
        # Save the model
        joblib.dump(self, file_path)
        print(f"Model saved to {file_path}")

    @classmethod
    def load_model(cls, filename):
        """
        Load a saved instance of MultiModalRegressor from a file in the mmr folder.
        """
        import joblib
        # Get the absolute path of the mmr folder
        mmr_folder = Path(__file__).parent
        # Create the full path for the file
        file_path = mmr_folder / filename
        # Load the model
        model_instance = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model_instance