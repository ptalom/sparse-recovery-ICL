import math
import numpy as np

import torch

from sparse_recovery.compressed_sensing import create_signal, create_Fourier_basis, create_normal_basis, create_orthonormal_basis

from sparse_recovery.matrix_factorization import get_matrices_UV



def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "compressed_sensing": CompressedSensing,
        "matrix_factorization": MatrixFactorization,
    }
    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError

class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class CompressedSensing(Task):
    def __init__(self, n_dims=None, batch_size=None, pool_dict=None,
                 N=50, d=20, sparsity=3, Phi=None, tau=0, variance=None, seed=None, **kwargs):
        super().__init__(n_dims, batch_size, pool_dict, **kwargs) 
        self.N = N
        self.d = d
        self.sparsity = sparsity
        self.tau = tau
        self.variance = variance
        self.seed = seed
        self.scale = 1
        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must in [0,1], received {self.tau}")

        # Construction de la base Phi
        if Phi is None or Phi == "identity":
            self.Phi = np.eye(d, dtype=np.float32)
        elif isinstance(Phi, str):
            Phi = Phi.lower()
            if Phi == "fourier":
                self.Phi = np.array(create_Fourier_basis(d), dtype=np.float32)
            elif Phi == "normal":
                self.Phi = np.array(create_normal_basis(d, seed=seed), dtype=np.float32)
            elif Phi == "orthonormal":
                self.Phi = np.array(create_orthonormal_basis(d, seed=seed), dtype=np.float32)
            else:
                raise ValueError(f"Unknown Phi type: {Phi}")
        else:
            self.Phi = np.array(Phi, dtype=np.float32)

        # Génération du signal sparse a*
        a_star, _ = create_signal(
            n=self.d, s=self.sparsity,
            distribution="normal", Phi=None,
            scaler=None, seed=self.seed
        )
        a_star = np.array(a_star, dtype=np.float32)

        # Génération de w* = Phi @ a*
        w_star = self.Phi @ a_star   # shape (d,)

        # 👇 Corrigé : w_b a la forme (1, d, 1)
        self.w_b = torch.tensor(w_star, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        #ys_b = ys_b.squeeze(-1)

        return ys_b



    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class MatrixFactorization(Task):
    
    def __init__(
        self,
        N: int,
        n1: int,
        n2: int,
        rank: int,
        problem: str = "matrix-completion",   # 'matrix-completion' / 'matrix-sensing'
        tau: float = 0.0,                     
        variance=None,
        seed: int | None = None,
        device: str = "cpu",
    ):
        assert problem in ("matrix-completion", "matrix-sensing"), \
            f"problem must be 'matrix-completion' or 'matrix-sensing', received :{problem}"
        assert 0.0 <= tau <= 1.0, f"tau must be in [0,1], received {tau}"

        self.N = N
        self.n1 = n1
        self.n2 = n2
        self.rank = rank
        self.problem = problem
        self.tau = tau
        self.variance = variance
        self.seed = seed
        self.device = device

        A_star, U_star, Sigma_star, V_star = get_matrices_UV(
            n_1=n1, n_2=n2, rank=rank, seed=seed
        )
        self.A_star = A_star.astype(np.float32)
        self.U_star = U_star.astype(np.float32)
        self.V_star = V_star.astype(np.float32)
        
        self.Sigma_star = Sigma_star.astype(np.float32)

        print(f"[Matrix Factorization Sampler] A*: {self.A_star.shape}, U*: {self.U_star.shape}, V*: {self.V_star.shape}")
        print(f"[Matrix Factorization Sampler] problem={self.problem}, N={self.N}, tau={self.tau}")
