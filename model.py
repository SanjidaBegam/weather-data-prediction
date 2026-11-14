
import numpy as np
from sklearn.linear_model import LinearRegression


def train_model(temperatures):

    temperatures = np.array(temperatures)

    days = np.arange(
        len(temperatures)
    ).reshape(-1, 1)

    model = LinearRegression()

    model.fit(days, temperatures)

    return model


def predict_temperature(model, total_days, future_days):

    days = np.arange(
        total_days,
        total_days + future_days
    ).reshape(-1, 1)

    predictions = model.predict(days)

    return predictions
