import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ==============================================
# ✅ CONFIGURATION CLASS
# ==============================================
@dataclass
class DataTransformationConfig:
    """
    Configuration class to store file paths or constants 
    used during the data transformation process.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# ==============================================
# ✅ DATA TRANSFORMATION CLASS
# ==============================================
class DataTransformation:
    """
    Handles all steps related to feature preprocessing:
    - Handling missing values
    - Scaling numerical columns
    - Encoding categorical columns
    - Saving the preprocessing pipeline
    """

    def __init__(self):
        # Initialize configuration
        self.data_transformation_config = DataTransformationConfig()

        # Ensure the directory for the preprocessor file exists
        os.makedirs(
            os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
            exist_ok=True
        )

    def get_data_transformer_object(self):
        """
        Create preprocessing pipelines for numerical and categorical columns.

        Returns:
            ColumnTransformer: A combined object containing preprocessing logic.
        """
        try:
            # Defining numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns: Median imputation + Standard scaling
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Pipeline for categorical columns: Mode imputation + One-hot encoding + Scaling
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Perform preprocessing on training and test data, and save the preprocessor.

        Args:
            train_path (str): Path to training CSV file
            test_path (str): Path to testing CSV file

        Returns:
            tuple: (transformed train array, transformed test array, preprocessor file path)
        """
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data successfully.")
            logging.info("Creating preprocessing object...")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on training and testing dataframes...")

            # Fit-transform on training data, transform on test data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features with the target column
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save the preprocessor object for later use
            logging.info("Saving preprocessing object as pickle file...")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info(f"Preprocessor saved at: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
        

# ==============================================
# ✅ MAIN EXECUTION BLOCK
# ==============================================
# if __name__ == "__main__":
#     try:
#         # Step 1: Import and run DataIngestion
#         from src.components.data_ingestion import DataIngestion
#         ingestion = DataIngestion()
#         train_path, test_path = ingestion.initiate_data_ingestion()

#         # Step 2: Run data transformation
#         transformation = DataTransformation()
#         train_arr, test_arr, path = transformation.initiate_data_transformation(train_path, test_path)

#         print("✅ Data transformation completed successfully!")
#         print(f"📦 Preprocessor saved at: {path}")

#     except Exception as e:
#         raise CustomException(e, sys)
