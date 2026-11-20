"""
GA-KELM: Genetic Algorithm optimised Kernel Extreme Learning Machine.

Stage 3 of the methodology — secondary model and hybrid fallback component.

References
----------
Huang G-B et al. (2006) Extreme Learning Machine.
Vapnik (1998) Statistical Learning Theory (kernel trick basis).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Kernel Extreme Learning Machine (supports multi-output regression)
# ---------------------------------------------------------------------------

class KELM:
    """
    Kernel Extreme Learning Machine for single or multi-horizon regression.

    Closed-form solution:
        beta = (K + I/C)^{-1} Y
    Prediction:
        Y_hat = K_test @ beta

    where K[i,j] = exp(-gamma * ||x_i - x_j||^2)   (RBF kernel)
    """

    def __init__(self, C: float = 1.0, gamma: float = 0.01):
        self.C = C
        self.gamma = gamma
        self.beta_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None
        self._is_1d: bool = False

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix: shape (n1, n2)."""
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2.0 * (X1 @ X2.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KELM":
        """Fit KELM: solve (K + I/C) beta = y."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            self._is_1d = True
        else:
            self._is_1d = False

        self.X_train_ = X
        n = X.shape[0]
        K = self._rbf_kernel(X, X)
        A = K + (np.eye(n, dtype=np.float64) / max(self.C, 1e-6))
        self.beta_ = np.linalg.solve(A, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using fitted KELM."""
        if self.beta_ is None or self.X_train_ is None:
            raise RuntimeError("KELM not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        K_test = self._rbf_kernel(X, self.X_train_)
        preds = K_test @ self.beta_
        preds = np.maximum(preds, 0.0)  # PM2.5 >= 0
        return preds.ravel() if self._is_1d else preds


# ---------------------------------------------------------------------------
# Genetic Algorithm optimiser for KELM hyperparameters
# ---------------------------------------------------------------------------

class GAKELMOptimizer:
    """
    Genetic Algorithm to search optimal (C, gamma) for KELM.

    Searched in log-10 space to cover orders of magnitude.
    """

    def __init__(
        self,
        pop_size: int = 15,
        n_generations: int = 15,
        mutation_rate: float = 0.25,
        mutation_sigma: float = 0.5,
        crossover_rate: float = 0.7,
        tournament_k: int = 3,
        C_range: Tuple[float, float] = (-1.0, 4.0),     # 0.1 to 10000
        gamma_range: Tuple[float, float] = (-4.0, 1.0), # 1e-4 to 10
        random_state: int = 42,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.C_range = C_range
        self.gamma_range = gamma_range
        self.rng = np.random.RandomState(random_state)
        self.history_: list = []

    def _create_population(self) -> np.ndarray:
        log_C = self.rng.uniform(*self.C_range, size=self.pop_size)
        log_gam = self.rng.uniform(*self.gamma_range, size=self.pop_size)
        return np.stack([log_C, log_gam], axis=1)

    def _decode(self, individual: np.ndarray) -> Tuple[float, float]:
        C = 10.0 ** np.clip(individual[0], *self.C_range)
        gamma = 10.0 ** np.clip(individual[1], *self.gamma_range)
        return float(C), float(gamma)

    def _fitness(
        self,
        individual: np.ndarray,
        X_tr: np.ndarray, y_tr: np.ndarray,
        X_va: np.ndarray, y_va: np.ndarray,
    ) -> float:
        try:
            C, gamma = self._decode(individual)
            kelm = KELM(C=C, gamma=gamma)
            kelm.fit(X_tr, y_tr)
            preds = kelm.predict(X_va)
            mae = float(np.mean(np.abs(preds - y_va)))
            if not np.isfinite(mae):
                return -1e9
            return -mae
        except Exception:
            return -1e9

    def _tournament_select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        idx = self.rng.choice(len(pop), size=self.tournament_k, replace=False)
        best = idx[np.argmax(fitness[idx])]
        return pop[best].copy()

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() < self.crossover_rate:
            mask = self.rng.randint(0, 2, size=p1.shape).astype(bool)
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
            return c1, c2
        return p1.copy(), p2.copy()

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        ind = individual.copy()
        for i in range(len(ind)):
            if self.rng.random() < self.mutation_rate:
                ind[i] += self.rng.normal(0.0, self.mutation_sigma)
        ind[0] = np.clip(ind[0], *self.C_range)
        ind[1] = np.clip(ind[1], *self.gamma_range)
        return ind

    def optimize(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
    ) -> Tuple[float, float]:
        logger.info(f"GA-KELM: Starting GA optimization (pop={self.pop_size}, gen={self.n_generations})")
        pop = self._create_population()
        fitness = np.array([
            self._fitness(ind, X_train, y_train, X_val, y_val)
            for ind in pop
        ])

        best_idx = np.argmax(fitness)
        best_ind = pop[best_idx].copy()
        best_fit = fitness[best_idx]

        for gen in range(self.n_generations):
            new_pop = [best_ind.copy()]  # Elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(pop, fitness)
                p2 = self._tournament_select(pop, fitness)
                c1, c2 = self._crossover(p1, p2)
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                new_pop.extend([c1, c2])

            pop = np.array(new_pop[:self.pop_size])
            fitness = np.array([
                self._fitness(ind, X_train, y_train, X_val, y_val)
                for ind in pop
            ])

            gen_best_idx = np.argmax(fitness)
            if fitness[gen_best_idx] > best_fit:
                best_fit = fitness[gen_best_idx]
                best_ind = pop[gen_best_idx].copy()

            self.history_.append({"generation": gen, "best_mae": -best_fit})
            if gen % 3 == 0 or gen == self.n_generations - 1:
                logger.info(f"  Gen {gen+1}/{self.n_generations} - Best val MAE: {-best_fit:.4f}")

        C, gamma = self._decode(best_ind)
        logger.info(f"GA-KELM: Optimal parameters C={C:.4f}, gamma={gamma:.6f} (val MAE: {-best_fit:.4f})")
        return C, gamma


# ---------------------------------------------------------------------------
# Scikit-learn Compatible Multi-Output GA-KELM Wrapper
# ---------------------------------------------------------------------------

class GAKELMModel:
    """
    GA-KELM Model compatible with Scikit-Learn API (fit, predict, save, load).
    Supports multi-output target horizons.
    """

    def __init__(
        self,
        pop_size: int = 15,
        n_generations: int = 15,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.7,
        random_state: int = 42,
        subsample: int = 6000,
        horizons: Optional[List[int]] = None,
        C: Optional[float] = None,
        gamma: Optional[float] = None,
    ):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.random_state = random_state
        self.subsample = subsample
        self.horizons = horizons or [1, 6, 12, 24, 48, 72]
        self.name = "ga_kelm"

        self.best_C_: Optional[float] = C
        self.best_gamma_: Optional[float] = gamma
        self.kelm_: Optional[KELM] = None
        self.feature_names_: Optional[List[str]] = None
        self.target_names_: Optional[List[str]] = None
        self._mu: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self.fitted: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.DataFrame, np.ndarray]) -> "GAKELMModel":
        """
        Fit GA-KELM on training data X, y.
        X: (n_samples, n_features)
        y: (n_samples, n_horizons)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.fillna(0.0).values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
            y_arr = y.fillna(0.0).values.astype(np.float64)
        else:
            y_arr = np.asarray(y, dtype=np.float64)

        n_samples = len(X_arr)
        rng = np.random.RandomState(self.random_state)

        # Standardise features based on training subset
        sub_n = min(self.subsample, n_samples)
        sub_idx = rng.choice(n_samples, size=sub_n, replace=False)
        X_train_sub = X_arr[sub_idx]
        y_train_sub = y_arr[sub_idx]

        self._mu = X_train_sub.mean(axis=0)
        self._std = X_train_sub.std(axis=0) + 1e-7

        X_train_scaled = (X_train_sub - self._mu) / self._std

        # Run GA optimization if C and gamma are not preset
        if self.best_C_ is None or self.best_gamma_ is None:
            ga_n = min(1200, sub_n)
            ga_idx = rng.choice(sub_n, size=ga_n, replace=False)
            val_split = int(0.75 * ga_n)

            X_ga = X_train_scaled[ga_idx]
            y_ga = y_train_sub[ga_idx]

            X_ga_tr = X_ga[:val_split]
            y_ga_tr = y_ga[:val_split]
            X_ga_va = X_ga[val_split:]
            y_ga_va = y_ga[val_split:]

            ga_opt = GAKELMOptimizer(
                pop_size=self.pop_size,
                n_generations=self.n_generations,
                mutation_rate=self.mutation_rate,
                crossover_rate=self.crossover_rate,
                random_state=self.random_state,
            )
            self.best_C_, self.best_gamma_ = ga_opt.optimize(
                X_ga_tr, y_ga_tr, X_ga_va, y_ga_va
            )

        # Fit final KELM with optimal hyperparameters
        self.kelm_ = KELM(C=self.best_C_, gamma=self.best_gamma_)
        self.kelm_.fit(X_train_scaled, y_train_sub)
        self.fitted = True
        logger.info(f"GAKELMModel successfully fitted on {sub_n} instances with C={self.best_C_:.4f}, gamma={self.best_gamma_:.6f}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict across all forecast horizons."""
        if not self.fitted or self.kelm_ is None:
            raise RuntimeError("GAKELMModel is not fitted yet.")
        if isinstance(X, pd.DataFrame):
            X_arr = X.fillna(0.0).values.astype(np.float64)
        else:
            X_arr = np.asarray(X, dtype=np.float64)

        X_scaled = (X_arr - self._mu) / self._std
        return self.kelm_.predict(X_scaled)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"GAKELMModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "GAKELMModel":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"GAKELMModel loaded from {path}")
        return obj
