"""
═══════════════════════════════════════════════════════════════
ÉTAPE 5 — COMPILATION ET ENTRAÎNEMENT
═══════════════════════════════════════════════════════════════

  - Split train / val / test
  - StandardScaler sur X
  - DataLoader PyTorch (mini-batches shufflés)
  - Optimiseur : AdamW + weight decay
  - Scheduler : OneCycleLR (warm-up + cosine decay)
  - Loss : HuberLoss (robuste aux outliers caloriques)
  - Early stopping (patience configurable)
  - Sauvegarde modèle + scaler + historique de loss
"""

import os
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model import CalorieResNet
from .config import (
    MODEL_PATH, SCALER_PATH, REPORT_PATH,
    EPOCHS, BATCH_SIZE, LR, TEST_SIZE, VAL_SIZE, PATIENCE, N_FEATURES,
)


class CalorieTrainer:
    def __init__(self):
        self.model   = CalorieResNet(n_features=N_FEATURES)
        self.scaler  = StandardScaler()
        self.trained = False
        self.history = {"train_loss": [], "val_loss": []}
        self._try_load()

    # ── Entraînement ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Lance l'entraînement complet et retourne les métriques d'évaluation.
        """
        t0 = time.time()

        print(f"[trainer] Architecture : {self.model.count_parameters():,} paramètres")

        # ── Split ─────────────────────────────────────────────────────────────
        # train / temp  puis  val / test sur temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(TEST_SIZE + VAL_SIZE), random_state=42
        )
        val_frac = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_frac), random_state=42
        )
        print(f"[trainer] Split → train={len(X_train):,} "
              f"| val={len(X_val):,} | test={len(X_test):,}")

        # ── Scaling ───────────────────────────────────────────────────────────
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        # ── Tenseurs & DataLoader ─────────────────────────────────────────────
        Xt  = torch.tensor(X_train_s, dtype=torch.float32)
        yt  = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
        Xv  = torch.tensor(X_val_s,   dtype=torch.float32)
        yv  = torch.tensor(y_val,     dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(
            TensorDataset(Xt, yt),
            batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        )

        # ── Optimiseur + Scheduler + Loss ─────────────────────────────────────
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr      = LR * 10,
            epochs      = EPOCHS,
            steps_per_epoch = len(loader),
            pct_start   = 0.3,
        )
        loss_fn = nn.HuberLoss(delta=200.0)   # tolérant aux outliers ±200 kcal

        # ── Boucle d'entraînement ─────────────────────────────────────────────
        best_val_loss = float("inf")
        best_state    = None
        wait          = 0

        for epoch in range(EPOCHS):
            # Train
            self.model.train()
            train_loss = 0.0
            for xb, yb in loader:
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
            train_loss /= len(loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(Xv)
                val_loss = loss_fn(val_pred, yv).item()

            self.history["train_loss"].append(round(train_loss, 4))
            self.history["val_loss"].append(round(val_loss, 4))

            if (epoch + 1) % 25 == 0:
                print(f"[trainer] Epoch {epoch+1:3d}/{EPOCHS} "
                      f"| Train loss {train_loss:8.1f} "
                      f"| Val loss {val_loss:8.1f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f"[trainer] Early stopping à l'epoch {epoch+1} "
                          f"(patience={PATIENCE})")
                    break

        # Restaure le meilleur état
        if best_state:
            self.model.load_state_dict(best_state)

        self.trained  = True
        elapsed       = round(time.time() - t0, 1)

        # ── Sauvegarde + rapport ──────────────────────────────────────────────
        self._save()
        report = {
            "training_time_s": elapsed,
            "epochs_run":      epoch + 1,
            "n_train":         len(X_train),
            "n_val":           len(X_val),
            "n_test":          len(X_test),
            "X_test":          X_test.tolist(),
            "y_test":          y_test.tolist(),
        }
        print(f"[trainer] ✓ Entraînement terminé en {elapsed}s")
        return report

    # ── Persistance ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)

    def _try_load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            try:
                self.model.load_state_dict(
                    torch.load(MODEL_PATH, weights_only=True, map_location="cpu")
                )
                self.model.eval()
                with open(SCALER_PATH, "rb") as f:
                    self.scaler = pickle.load(f)
                self.trained = True
                print("[trainer] ✓ Modèle pré-entraîné chargé depuis le disque")
            except Exception as e:
                print(f"[trainer] Impossible de charger : {e}")