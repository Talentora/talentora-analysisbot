import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from test.job_processor import JobProcessor

class MultiModalRegressor:
    def __init__(self, data, target, modalities=['facial', 'prosody', 'language'], 
                 svr_params=None, meta_params=None, num_top_features=6, k_fold_splits = 6):
        """
        Parameters:
        - data: dict where keys are modality names and values are feature matrices (DataFrames or numpy arrays)
        - target: array-like of target values
        - modalities: list of modality names (e.g., ['facial', 'prosody', 'language'])
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
            'svr__C': [ 0.1, 0.2, 0.8, 0.9, 1, 10],
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
            ridge = Ridge(alpha=0.36)
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
                y_train = y.iloc[train_index]
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
        """
        modality_preds = []
        for mod in self.modalities:
            X_new = new_data[mod]
            # Ensure the new data has the same columns as selected during training
            if isinstance(X_new, pd.DataFrame) and mod in self.selected_features:
                X_new = X_new[self.selected_features[mod]]
            pred = self.svr_models[mod].predict(X_new)
            modality_preds.append(pred)
        meta_features = np.column_stack(modality_preds)
        return self.meta_model.predict(meta_features)

    def evaluate(self, true_values, predictions):
        """
        Computes evaluation metrics for regression: MSE, MAE, and R^2 score.
        """
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
        return mse, mae, r2

# Example usage remains largely unchanged.
if __name__ == "__main__":
    print("Loading hume data")
    face_csv_path = "face_predictions_1.csv"
    pros_csv_path = "prosody_predictions_1.csv"
    lang_csv_path = "language_predictions_1.csv"
    face_csv_path_secondary = "face_predictions.csv"
    pros_csv_path_secondary = "prosody_predictions.csv"
    lang_csv_path_secondary = "language_predictions.csv"
    
    # Load CSV data into DataFrames
    job = JobProcessor()
    df_face, df_pros, df_lang = job.generate_dataframes_from_csv(face_csv_path, pros_csv_path, lang_csv_path, face_csv_path_secondary, pros_csv_path_secondary, lang_csv_path_secondary)
    
    # Load target labels
    labels = job.load_labels()
    
    # Merge each modality's data with labels
    XY_face, XY_pros, XY_lang = job.merge_data(labels, df_face, df_pros, df_lang)
    
    # Convert the merged DataFrames into model-compatible inputs:
    # model_data is a dict with keys for each modality and target is an array-like.
    model_data, target = job.generate_mmr_compatible_input(XY_face, XY_pros, XY_lang)
    
    modalities = ['facial', 'prosody', 'language']

    indices = np.arange(len(target))
    # 80% training, 20% testing
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create modality-specific training and test data dictionaries
    train_data = {mod: model_data[mod].iloc[train_idx] for mod in modalities}
    test_data  = {mod: model_data[mod].iloc[test_idx] for mod in modalities}    
    train_target = target[train_idx]
    test_target  = target[test_idx]
    
    # -------------------------------
    # Plot informative graphs about the labels (omitted here for brevity)
    # -------------------------------
    
    # Initialize the multimodal regressor using the training data only.
    mmr = MultiModalRegressor(train_data, train_target, modalities, k_fold_splits=6)
    
    # First, perform feature selection using Ridge regression.
    mmr.select_features_ridge()
    
    # Build and grid-search the SVR pipelines using the selected features.
    mmr.build_svr_pipeline()
    mmr.grid_search_svr()
    
    # Generate out-of-fold predictions using CV on the training data.
    mmr.generate_oof_predictions()
    
    # Train the meta-model (stacked ensemble) using the OOF predictions.
    mmr.train_meta_model()
    
    # Optionally, run grid search on the meta-model for additional tuning.
    mmr.grid_search_meta()
    
    # ---- Evaluation on Unseen Test Data ----
    # For each modality, get predictions on the test set using the trained SVR models.
    meta_features_test = np.column_stack([
        # Ensure we subset test data to the selected features.
        mmr.svr_models[mod].predict(test_data[mod][mmr.selected_features[mod]])
        for mod in modalities
    ])
    # Use the meta model to produce final predictions on the test data.
    test_predictions = mmr.meta_model.predict(meta_features_test)
    
    # Evaluate performance on the held-out test set.
    mmr.evaluate(test_target, test_predictions)
    
    # Plot true target vs. predicted values for the test data.
    plt.figure(figsize=(8, 8))
    plt.scatter(test_target, test_predictions, alpha=0.7, label='Predictions')
    
    # Plot the ideal line (y = x) for reference.
    min_val = min(test_target.min(), test_predictions.min())
    max_val = max(test_target.max(), test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    
    plt.xlabel('True Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Data: True vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()