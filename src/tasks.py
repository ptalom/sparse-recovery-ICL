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

        return ys_b


    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
    

class MatrixFactorization(Task):
    def __init__(self, n_dims=None, batch_size=None, pool_dict=None,
                 n1=20, n2=20, rank=5, N=50, tau=0.0, variance=None,
                 problem="matrix-completion", seed=None, target_rank=None, **kwargs):
        self.target_rank = target_rank
        if "target_rank" in kwargs:
            kwargs.pop("target_rank") 

        super().__init__(n_dims, batch_size, pool_dict, **kwargs)

        self.N = N
        self.n1 = n1
        self.n2 = n2
        self.rank = rank if target_rank is None else target_rank
        self.tau = tau
        self.variance = variance
        self.problem = problem
        self.seed = seed

        if not (0.0 <= self.tau <= 1.0):
            raise ValueError(f"tau must in [0,1], reçu {self.tau}")
        #print(f"[DEBUG TASK] A_star min={self.A_star.min()}, max={self.A_star.max()}, shape={self.A_star.shape}")

        self.A_star, self.U_star, self.Sigma_star, self.V_star = get_matrices_UV(
            n1, n2, self.rank, symmetric=False, normalize=True, scale=None, seed=seed
        )

        self.A_b = torch.tensor(self.A_star, dtype=torch.float32)
        self.U_b = torch.tensor(self.U_star, dtype=torch.float32)
        self.V_b = torch.tensor(self.V_star, dtype=torch.float32)
    

    def update_from_sampler(self, sampler):
        """
        Synchroniser la task avec le sampler qui génère réellement les données.
        Appeler ceci avant evaluation si le sampler peut changer (différentes dims).
        """
        # copier matrices / paramètres depuis le sampler (sampler vient de get_data_sampler)
        self.A_star = getattr(sampler, "A_star", self.A_star)
        self.U_star = getattr(sampler, "U_star", self.U_star)
        self.V_star = getattr(sampler, "V_star", self.V_star)
        self.Sigma_star = getattr(sampler, "Sigma_star", getattr(self, "Sigma_star", None))

        # dims / autres paramètres
        self.n1 = int(getattr(sampler, "n1", self.n1))
        self.n2 = int(getattr(sampler, "n2", self.n2))
        self.N = int(getattr(sampler, "N", self.N))
        self.rank = getattr(sampler, "rank", self.rank)
        self.tau = getattr(sampler, "tau", self.tau)
        self.variance = getattr(sampler, "variance", self.variance)
        self.problem = getattr(sampler, "problem", self.problem)
        self.seed = getattr(sampler, "seed", self.seed)

        # mettre à jour les tensors si besoin
        self.A_b = torch.tensor(self.A_star, dtype=torch.float32)
        self.U_b = torch.tensor(self.U_star, dtype=torch.float32)
        self.V_b = torch.tensor(self.V_star, dtype=torch.float32)

  

    def evaluate(self, xs_b, sampler=None):
        if sampler is not None:
            self.update_from_sampler(sampler)

        X1 = xs_b[:, :, :self.n1]
        X2 = xs_b[:, :, self.n1:]

        if X1.shape[-1] != self.n1 or X2.shape[-1] != self.n2:
            raise ValueError(
                f"Shapes invalides: X1 {X1.shape}, X2 {X2.shape}, attendu n1={self.n1}, n2={self.n2}"
            )

        A_tensor = torch.as_tensor(self.A_star, dtype=torch.float32, device=xs_b.device)

        print(f"[DEBUG EVAL FIX] A_star shape={A_tensor.shape}, "
            f"X1 shape={X1.shape}, X2 shape={X2.shape}")

        ys_b = torch.einsum('bni,ij,bnj->bn', X1, A_tensor, X2)
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
