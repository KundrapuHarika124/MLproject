import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import(
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
) 
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from sklearn.neighbors import KNeighborsRegressor
from src.utils import save_object
@dataclass

class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        logging.info("Entered the model trainer component")
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Random Forest Regressor': RandomForestRegressor(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'AdaBoost Regressor': AdaBoostRegressor(),
                'SVR': SVR(),
                'KNeighbors Regressor': KNeighborsRegressor(),
                'XGB Regressor': XGBRegressor(),
                'CatBoost Regressor': CatBoostRegressor(verbose=False)
            }

            model_report: dict = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy")

            best_model = models[best_model_name]
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Best model found: {best_model_name} with R2 score: {r2_square}")
            return r2_square, best_model_name

        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predicted = model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            model_report[model_name] = r2_square
        return model_report