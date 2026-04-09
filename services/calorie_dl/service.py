from .data_loader import load_data
from .preprocessing import clean_data
from .feature_engineering import build_features
from .train import CalorieTrainer
from .inference import predict


class CalorieDLService:
    def __init__(self):
        self.trainer = CalorieTrainer()

    def train(self):
        df = load_data()
        df = clean_data(df)
        X, y = build_features(df)

        self.trainer.train(X, y)

    def predict(self, age, gender, weight, height, activity):
        if not self.trainer.trained:
            self.train()

        return predict(
            self.trainer.model,
            self.trainer.scaler,
            age, gender, weight, height, activity
        )