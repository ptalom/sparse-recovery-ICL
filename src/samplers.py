import numpy as np
import torch
import math

from sparse_recovery.compressed_sensing import create_signal, get_measures
from sparse_recovery.matrix_factorization import get_data_matrix_factorization, get_matrices_UV

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError

def get_data_sampler(data_name, n_dims, **kwargs):
    
    samplers = {
        "sparse_recovery": CompressedSensingSampler,
        "matrix_factorization": MatrixFactorizationSampler,
    }
    
    data_name = data_name.lower()
    if data_name in samplers:
        sampler_cls = samplers[data_name]
        return sampler_cls(n_dims=n_dims, **kwargs)
    
    print(f"Unknown sampler: {data_name}")
    raise NotImplementedError



class CompressedSensingSampler(DataSampler):
    def __init__(self, n_dims, **kwargs):
        super().__init__(n_dims)
        self.N = kwargs.get("N", 50)
        self.d = kwargs.get("d", n_dims)
        self.s = kwargs.get("s", 5)
        self.tau = kwargs.get("tau", 0.5)
        self.variance = kwargs.get("variance", None)
        self.seed = kwargs.get("seed", None)
        self.Phi = kwargs.get("Phi", np.eye(self.d))  # base par défaut = identité

    def sample_xs(self, n_points, batch_size=1, n_dims_truncated=None, **kwargs):
        if n_points is None:
            n_points = self.N  

        xs_list = []

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

        #print("xs shape = ", xs.shape)
        return xs

class MatrixFactorizationSampler(DataSampler):
    def __init__(self, n_dims, **kwargs):
        super().__init__(n_dims)
        
        self.N = int(kwargs.get("N", 50))            
        self.n1 = int(kwargs.get("n1", n_dims))      
        self.n2 = int(kwargs.get("n2", n_dims))      
        self.rank = int(kwargs.get("rank", 3))       
        self.tau = float(kwargs.get("tau", 0.0))     
        self.variance = kwargs.get("variance", None)
        self.seed = kwargs.get("seed", None)
        self.problem = kwargs.get("problem", "matrix-completion")  

        # Génération de la matrice cible A* = U Σ V^T
        self.A_star, self.U_star, self.Sigma_star, self.V_star = get_matrices_UV(
            self.n1, self.n2, self.rank, seed=self.seed, symmetric=False, normalize=True, scale=None
        )
        
        # Debug info
        print(f"[DEBUG] Sampler init: N={self.N}, n1={self.n1}, n2={self.n2}, rank={self.rank}, tau={self.tau}")
        print(f"[DEBUG] Shapes: A_star={self.A_star.shape}, U_star={self.U_star.shape}, V_star={None if self.V_star is None else self.V_star.shape}")
        print(f"[DEBUG SAMPLER] A_star min={self.A_star.min()}, max={self.A_star.max()}, shape={self.A_star.shape}")

    def sample_xs(self, n_points=None, batch_size=1, n_dims_truncated=None, seeds=None, **kwargs):
        if n_points is None:
            n_points = self.N

        xs_list = []

        for _ in range(batch_size):
            (X1, X2, _, _), _ = get_data_matrix_factorization(
                self.A_star,
                self.U_star,
                self.V_star,
                n_points,
                problem=self.problem,
                tau=self.tau,
                variance=self.variance,
                seed=self.seed,
            )
            X = np.concatenate([X1, X2], axis=-1).astype(np.float32)  # (n_points, n1+n2)
            xs_list.append(X)

        xs = torch.tensor(np.stack(xs_list), dtype=torch.float32)  # (batch, n_points, n1+n2)

        return xs
