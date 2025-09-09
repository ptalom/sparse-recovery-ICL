import numpy as np
import torch
import math

from sparse_recovery.compressed_sensing import create_signal, get_measures
from sparse_recovery.matrix_factorization import get_data_matrix_factorization, solve_matrix_factorization_nuclear_norm

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

def get_data_sampler(name, **kwargs):
    samplers = {
        "sparse_recovery": CompressedSensingSampler,
        "matrix_factorization": MatrixFactorizationSampler,
    }
    name = name.lower()
    if name in samplers:
        return samplers[name](**kwargs)
    raise ValueError(f"Unknown sampler name: {name}")



def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t

class CompressedSensingSampler(DataSampler):
    def __init__(self, n_dims, **kwargs):
        super().__init__(n_dims)
        # Enregistre les hyperparamètres qu’on va utiliser
        self.N = kwargs.get("N", 50)
        self.d = kwargs.get("d", n_dims)
        self.s = kwargs.get("s", 5)
        self.tau = kwargs.get("tau", 0.5)
        self.variance = kwargs.get("variance", None)
        self.seed = kwargs.get("seed", None)
        self.Phi = kwargs.get("Phi", np.eye(self.d))  # base par défaut = identité

    def sample_xs(self, n_points, batch_size=1, n_dims_truncated=None, **kwargs):
        if n_points is None:
            n_points = self.N  # valeur par défaut

        xs_list, ys_list, w_list, a_list = [], [], [], []

        for _ in range(batch_size):
            # Génération du signal sparse a*
            a_star, _ = create_signal(
                n=self.d, s=self.s,
                distribution="normal", Phi=None,
                scaler=None, seed=self.seed
            )
            a_star = np.array(a_star, dtype=np.float32)

            # Génération de w* = Phi @ a*
            w_star = self.Phi @ a_star

            # Génération de la matrice de mesures M
            M = get_measures(
                N=n_points,
                Phi=self.Phi,
                tau=self.tau,
                variance=self.variance,
                seed=self.seed,
            )

            # Assurez-vous que X est 2D : (N, d)
            X = np.array(M, dtype=np.float32)
            if X.ndim > 2:
                # Si X a plus de 2 dimensions, prenez seulement la première "slice"
                X = X[0]  # (n_points, d)

            # Couper/ajuster le nombre de points si nécessaire
            X = X[:n_points, :]  # assure que X a exactement n_points lignes

            # Calcul de y
            #y = (X @ w_star).reshape(-1, 1)  # (n_points, 1)

            xs_list.append(X)
            
        # Conversion en Tensors
        xs = torch.tensor(np.stack(xs_list), dtype=torch.float32)    # (batch_size, n_points, d)

        print("xs shape = ", xs.shape)
        return xs


class MatrixFactorizationSampler(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def sample_xs(self):
        
        # Mesures (X1, X2, X2•X1, y)
        (X1, X2, X2_bullet_X1, y_star), _ = get_data_matrix_factorization(
            self.A_star, self.U_star, self.V_star,
            self.N,
            problem=self.problem,
            tau=self.tau,
            variance=self.variance,
            seed=self.seed,
        )
        # X1: (N, n1)  | X2: (N, n2)  | y_star: (N,)

        X_np = np.concatenate([X1, X2], axis=1).astype(np.float32)
        y_np = y_star.astype(np.float32).reshape(-1)

        X = torch.tensor(X_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, N, n1+n2)
        y = torch.tensor(y_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, N)

        return X, y, None, None