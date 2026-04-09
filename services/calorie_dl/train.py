import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model import CalorieNet


class CalorieTrainer:
    def __init__(self):
        self.model = CalorieNet()
        self.scaler = StandardScaler()
        self.trained = False

    def train(self, X, y, epochs=50):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for _ in range(epochs):
            pred = self.model(X_train)
            loss = loss_fn(pred, y_train)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.trained = True
        return self