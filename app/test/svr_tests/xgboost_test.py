import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

from app.test.svr_tests.job_processor import JobProcessor

# NEW: import XGBoost
from xgboost import XGBRegressor

class MultiModalRegressor:
    def __init__(self, 
                 data, 
                 target, 
                 modalities=['facial', 'prosody', 'language'], 
                 use_xgboost=False,
                 svr_params=None, 
                 xgb_params=None,
                 meta_params=None, 
                 num_top_features=41,   # <-- updated to 41
                 k_fold_splits = 6):
        """
        Parameters:
        - data: dict where keys are modality names and values are feature matrices (DataFrames or numpy arrays)
        - target: array-like of target values
        - modalities: list of modality names (e.g., ['facial', 'prosody', 'language'])
        - use_xgboost: if True, use XGBoost as base regressor; if False, use SVR.
        - svr_params: dictionary of grid search parameters for the SVR models
        - xgb_params: dictionary of grid search parameters for the XGBoost models
        - meta_params: dictionary of grid search parameters for the meta model
        - num_top_features: number of top features to select based on Ridge regression coefficients
        - k_fold_splits: number of folds for KFold cross-validation
        """
        self.data = data
        self.target = target
        self.modalities = modalities
        self.use_xgboost = use_xgboost  
        self.num_top_features = num_top_features
        self.k_fold_splits = k_fold_splits

        # Default grid parameters for SVR models if none are provided
        self.svr_params = svr_params if svr_params is not None else {
            'regressor__C': [0.1, 0.2, 0.8, 0.9, 1, 10, 11, 12],  # <-- updated C values
            'regressor__gamma': ['scale', 'auto'],
            'regressor__kernel': ['rbf']
        }

        # Default grid parameters for XGBoost if none are provided
        self.xgb_params = xgb_params if xgb_params is not None else {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [3, 5],
            'regressor__learning_rate': [0.01, 0.1],
            'regressor__subsample': [0.8, 1.0]
        }

        # Default grid parameters for the meta model (using Linear Regression)
        self.meta_params = meta_params if meta_params is not None else {
            'meta__fit_intercept': [True, False]
        }

        # Hold the best base pipelines for each modality
        self.base_models = {}
        self.base_best_params = {}

        # Hold out-of-fold predictions for each modality
        self.oof_predictions = {mod: np.zeros(len(target)) for mod in modalities}

        # Meta model
        self.meta_model = None
        self.meta_best_params = None

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
            ridge = Ridge(alpha=0.19)   # <-- updated alpha
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

    def build_regressor_pipelines(self):
        """
        Constructs a pipeline for each modality using either SVR or XGBoost 
        (based on use_xgboost) with StandardScaler.
        """
        pipelines = {}
        for mod in self.modalities:
            if self.use_xgboost:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', XGBRegressor(
                        objective='reg:squarederror',
                        eval_metric='rmse',
                        # tree_method='gpu_hist', # if you have a GPU
                    ))
                ])
            else:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('regressor', SVR())
                ])

            pipelines[mod] = pipeline

        self.base_models = pipelines

    def grid_search_regressors(self):
        """
        Performs grid search with CV for each modality's pipeline (either SVR or XGBoost).
        """
        best_estimators = {}
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            pipeline = self.base_models[mod]

            kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)

            # Choose the param grid based on whether we're using XGBoost or SVR
            param_grid = self.xgb_params if self.use_xgboost else self.svr_params

            grid_search = GridSearchCV(pipeline, param_grid, cv=kf)
            grid_search.fit(X, y)
            best_estimators[mod] = grid_search.best_estimator_
            self.base_best_params[mod] = grid_search.best_params_

            print(f"Best params for {mod} modality "
                  f"({'XGB' if self.use_xgboost else 'SVR'}): {grid_search.best_params_}")

        self.base_models = best_estimators

    def generate_oof_predictions(self):
        """
        Generates out-of-fold (OOF) predictions using CV for each modality.
        Each prediction is from a model that did not see the sample during training.
        """
        kf = KFold(n_splits=self.k_fold_splits, shuffle=True, random_state=42)
        n_samples = len(self.target)

        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            preds = np.zeros(n_samples)

            for train_index, test_index in kf.split(X):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train = y.iloc[train_index]
                else:
                    X_train, X_test = X[train_index], X[test_index]
                    y_train = y[train_index]

                model = self.base_models[mod]
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

        self.meta_model = Pipeline([
            ('scaler', StandardScaler()),
            ('meta', LinearRegression())
        ])
        self.meta_model.fit(meta_features, y)

    def grid_search_meta(self):
        """
        Performs grid search with CV for the meta model based on the combined OOF predictions.
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
            pred = self.base_models[mod].predict(X_new)
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


if __name__ == "__main__":
    print("Loading hume data")

    # Updated CSV paths (single set, matching your new usage)
    face_csv_path = "face_predictions.csv"
    pros_csv_path = "prosody_predictions.csv"
    lang_csv_path = "language_predictions.csv"

    # If needed, you can still comment out or remove these second paths:
    # face_csv_path_secondary = "face_predictions.csv"
    # pros_csv_path_secondary = "prosody_predictions.csv"
    # lang_csv_path_secondary = "language_predictions.csv"

    job = JobProcessor()
    # Only reading one set of CSV files:
    df_face, df_pros, df_lang = job.generate_dataframes_from_csv(
        face_csv_path, 
        pros_csv_path, 
        lang_csv_path
        # Uncomment these if you want to include a second set:
        # face_csv_path_secondary, 
        # pros_csv_path_secondary, 
        # lang_csv_path_secondary
    )
    
    labels = job.load_labels()
    XY_face, XY_pros, XY_lang = job.merge_data(labels, df_face, df_pros, df_lang)
    model_data, target = job.generate_mmr_compatible_input(XY_face, XY_pros, XY_lang)
    
    modalities = ['facial', 'prosody', 'language']
    indices = np.arange(len(target))
    # Updated test size to 30%, matching your new usage
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
    
    train_data = {mod: model_data[mod].iloc[train_idx] for mod in modalities}
    test_data  = {mod: model_data[mod].iloc[test_idx] for mod in modalities}
    train_target = target[train_idx]
    test_target  = target[test_idx]
    
    # Initialize the regressor (set use_xgboost=True if you actually want XGBoost)
    mmr = MultiModalRegressor(
        data=train_data,
        target=train_target,
        modalities=modalities,
        use_xgboost=True,  # Change to True if you'd like to run XGB
        k_fold_splits=6
    )
    
    # Feature selection
    mmr.select_features_ridge()
    
    # Build pipelines (XGB if use_xgboost=True, otherwise SVR)
    mmr.build_regressor_pipelines()
    
    # Grid search the base models
    mmr.grid_search_regressors()
    
    # Generate out-of-fold predictions
    mmr.generate_oof_predictions()
    
    # Train meta model (stacked)
    mmr.train_meta_model()
    
    # Optionally grid search the meta model
    mmr.grid_search_meta()
    
    # Evaluate on test set
    meta_features_test = np.column_stack([
        mmr.base_models[mod].predict(test_data[mod][mmr.selected_features[mod]])
        for mod in modalities
    ])
    test_predictions = mmr.meta_model.predict(meta_features_test)
    
    mmr.evaluate(test_target, test_predictions)

    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(test_target, test_predictions, alpha=0.7, label='Predictions')
    min_val = min(test_target.min(), test_predictions.min())
    max_val = max(test_target.max(), test_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal')
    plt.xlabel('True Target Values')
    plt.ylabel('Predicted Values')
    plt.title('Test Data: True vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()
