import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from test.job_processor import JobProcessor

class MultiModalRegressor:
    def __init__(self, data, target, modalities = ['facial', 'prosody', 'language'], svr_params=None, meta_params=None):
        """
        Parameters:
        - data: dict where keys are modality names and values are feature matrices (numpy arrays or pandas DataFrames)
        - target: array-like of target values
        - modalities: list of modality names (e.g., ['facial', 'prosody', 'language'])
        - svr_params: dictionary of grid search parameters for the SVR models (default provided if None)
        - meta_params: dictionary of grid search parameters for the meta model (default provided if None)
        """
        self.data = data
        self.target = target
        self.modalities = modalities

        # Default grid parameters for SVR models if none are provided
        self.svr_params = svr_params if svr_params is not None else {
            'svr__C': [0.1, 1, 10],
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

    def build_svr_pipeline(self):
        """Constructs an SVR pipeline for each modality."""
        pipelines = {}
        for mod in self.modalities:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR())
            ])
            pipelines[mod] = pipeline
        self.svr_models = pipelines

    def grid_search_svr(self):
        """Performs grid search with LOOCV for each modality's SVR pipeline."""
        best_estimators = {}
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            pipeline = self.svr_models[mod]
            # Using Leave-One-Out Cross-Validation
            grid_search = GridSearchCV(pipeline, self.svr_params, cv=LeaveOneOut())
            grid_search.fit(X, y)
            best_estimators[mod] = grid_search.best_estimator_
            self.svr_best_params[mod] = grid_search.best_params_
            print(f"Best params for {mod} modality: {grid_search.best_params_}")
        self.svr_models = best_estimators

    def generate_oof_predictions(self):
        """
        Generates out-of-fold (OOF) predictions using LOOCV for each modality.
        Each data point's prediction is obtained from a model that did not see it during training.
        """
        loo = LeaveOneOut()
        n_samples = len(self.target)
        for mod in self.modalities:
            X = self.data[mod]
            y = self.target
            preds = np.zeros(n_samples)
            for train_index, test_index in loo.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train = y[train_index]
                model = self.svr_models[mod]
                model.fit(X_train, y_train)
                preds[test_index] = model.predict(X_test)
            self.oof_predictions[mod] = preds
        return self.oof_predictions

    def train_meta_model(self):
        """
        Trains the meta model using the out-of-fold predictions from each modality.
        The meta model combines the predictions from each SVR into a final output.
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
        Performs grid search with LOOCV for the meta model.
        This finds the best parameters for the meta model based on the combined modality predictions.
        """
        meta_features = np.column_stack([self.oof_predictions[mod] for mod in self.modalities])
        y = self.target
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('meta', LinearRegression())
        ])
        grid_search = GridSearchCV(pipeline, self.meta_params, cv=LeaveOneOut())
        grid_search.fit(meta_features, y)
        self.meta_model = grid_search.best_estimator_
        self.meta_best_params = grid_search.best_params_
        print(f"Best params for meta model: {grid_search.best_params_}")

    def predict(self, new_data):
        """
        Predicts the target value for new data.
        new_data: dict with keys corresponding to modalities and values as feature matrices.
        Returns the final prediction from the meta model.
        """
        modality_preds = []
        for mod in self.modalities:
            X_new = new_data[mod]
            pred = self.svr_models[mod].predict(X_new)
            modality_preds.append(pred)
        meta_features = np.column_stack(modality_preds)
        return self.meta_model.predict(meta_features)

    def evaluate(self, true_values, predictions):
        """
        Computes evaluation metrics for regression.
        Returns MSE, MAE, and R^2 score.
        """
        mse = mean_squared_error(true_values, predictions)
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        print(f"MSE: {mse}, MAE: {mae}, R2: {r2}")
        return mse, mae, r2

# Example usage:
if __name__ == "__main__":
    # For demonstration, generate some random data.
    face_csv_path = "face_predictions.csv"
    pros_csv_path = "prosody_predictions.csv"
    lang_csv_path = "language_predictions.csv"
    
    job = JobProcessor()
    df_face, df_pros, df_lang = job.generate_dataframes_from_csv(face_csv_path, pros_csv_path, lang_csv_path)
    labels = job.load_labels()
    XY_face, XY_pros, XY_lang = job.merge_data(labels, df_face, df_pros, df_lang)
    data = job.generate_mmr_compatible_input(XY_face, XY_pros, XY_lang)
    
    
    
    n_samples = 17  # You have 17 videos/data points
    np.random.seed(42)
    data = {
        'facial': np.random.rand(n_samples, 10),   # 10 features for facial modality
        'prosody': np.random.rand(n_samples, 8),     # 8 features for prosody modality
        'language': np.random.rand(n_samples, 12)    # 12 features for language modality
    }
    target = np.random.rand(n_samples) * 9 + 1  # Random target values between 1 and 10
    modalities = ['facial', 'prosody', 'language']

    # Initialize the multimodal regressor
    mmr = MultiModalRegressor(data, target, modalities)
    mmr.build_svr_pipeline()
    
    # Run grid search for SVR models for each modality
    mmr.grid_search_svr()
    
    # Generate out-of-fold predictions using LOOCV
    mmr.generate_oof_predictions()
    
    # Train the meta-model using the OOF predictions
    mmr.train_meta_model()
    
    # Optionally run grid search on the meta-model
    mmr.grid_search_meta()
    
    # Evaluate on the training data (in practice, use a separate test set)
    meta_features = np.column_stack([mmr.oof_predictions[mod] for mod in modalities])
    predictions = mmr.meta_model.predict(meta_features)
    mmr.evaluate(target, predictions)
    
    # Example prediction on new data
    new_data = {
        'facial': np.random.rand(1, 10),
        'prosody': np.random.rand(1, 8),
        'language': np.random.rand(1, 12)
    }
    final_prediction = mmr.predict(new_data)
    print("Final Prediction for new data:", final_prediction)