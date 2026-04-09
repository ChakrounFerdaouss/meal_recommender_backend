import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate(model, scaler, X, y):
    """
    Évalue un modèle PyTorch sur dataset classique
    Retourne MAE + R²
    """

    # ── Scale ──
    X_scaled = scaler.transform(X)

    # ── Tensor ──
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model.eval()

    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten()

    y = np.array(y).flatten()

    return {
        "mae": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }