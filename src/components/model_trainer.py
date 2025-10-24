import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# =====================================================
# ✅ Configuration class to define model path
# =====================================================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


# =====================================================
# ✅ ModelTrainer Class
# =====================================================
class ModelTrainer:
    def __init__(self):
        # Use configuration class
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models, evaluates them,
        and saves the best-performing model as a pickle file.
        """
        try:
            logging.info("Splitting training and testing input data")

            # Split the features and target
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # =====================================================
            # ✅ Define models to compare
            # =====================================================
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # =====================================================
            # ✅ Define hyperparameters for tuning
            # =====================================================
            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBoost": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.01, 0.5, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            # =====================================================
            # ✅ Evaluate all models
            # =====================================================
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            # Find the best model and score
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(f"Best Model: {best_model_name} | R² Score: {best_model_score}")

            # =====================================================
            # ✅ Save the best model
            # =====================================================
            if best_model_score < 0.6:
                raise CustomException("No good model found — R² < 0.6")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully!")

            # Evaluate final R² on test data
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Final Test R² Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


# =====================================================
# ✅ Main entry point for testing this module
# =====================================================
# if __name__ == "__main__":
#     try:
#         from src.components.data_ingestion import DataIngestion
#         from src.components.data_transformation import DataTransformation

#         logging.info("Starting model training pipeline...")

#         # 1️⃣ Data Ingestion
#         ingestion = DataIngestion()
#         train_path, test_path = ingestion.initiate_data_ingestion()

#         # 2️⃣ Data Transformation
#         transformation = DataTransformation()
#         train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)

#         # 3️⃣ Model Training
#         trainer = ModelTrainer()
#         r2 = trainer.initiate_model_trainer(train_arr, test_arr)

#         print(f"\n✅ Model Training Completed Successfully!")
#         print(f"📊 Final R² Score on Test Data: {r2:.4f}")

#     except Exception as e:
#         raise CustomException(e, sys)
