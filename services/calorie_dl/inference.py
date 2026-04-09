import torch


def predict(model, scaler, age, gender, weight, height, activity):
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == 1 else -161)

    x = [[age, gender, weight, height, activity, bmr]]
    x = scaler.transform(x)

    x = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        return float(model(x).item())