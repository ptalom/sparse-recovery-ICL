import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np

from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

from sparse_recovery.matrix_factorization import F_cvxpy


def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    else:
        raise NotImplementedError

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "compressed_sensing": [
            (LeastSquaresModel, {}),
            (L1MinimizationModel, {"epsilon": 1e-6})
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001]],
        "matrix_factorization": [
            (LeastSquaresModel, {}),
            (NuclearNormMinimizationModel, {"epsilon": 1e-6}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, max_n_dims=None):
        super(TransformerModel, self).__init__()
        self.n_dims = n_dims
        self.max_n_dims = max_n_dims if max_n_dims is not None else n_dims
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(self.max_n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        #print("DEBUG xs_b type:", type(xs_b), "shape:", xs_b.shape)
        #print("DEBUG ys_b type:", type(ys_b), "shape:", ys_b.shape)
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        # DEBUG prints
        #print("DEBUG forward xs:", xs.shape)
        #print("DEBUG forward ys:", ys.shape)

        if xs.shape[2] > self.max_n_dims:
            xs_pad = xs[:, :, :self.max_n_dims]
        elif xs.shape[2] < self.max_n_dims:
            pad_size = self.max_n_dims - xs.shape[2]
            xs_pad = torch.cat([xs, torch.zeros(xs.shape[0], xs.shape[1], pad_size, device=xs.device)], dim=2)
        else:
            xs_pad = xs

        zs = self._combine(xs_pad, ys)

        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)

        if inds is None:
            inds = torch.arange(xs.shape[1], device=xs.device)
        else:
            inds = torch.tensor(inds, device=xs.device)
            if max(inds) >= xs.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        # On ne prédit que sur les indices originaux
        return prediction[:, ::2, 0][:, inds]




# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])

        preds = []
        for i in inds:
            pred = torch.zeros_like(ys[:, 0], dtype=torch.float32)
            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # Convert to numpy
                    X_np = train_xs.numpy().astype("float32")
                    y_np = train_ys.numpy().astype("float32")

                    # Fallback si trop peu de points
                    if X_np.shape[0] < 2:
                        w_pred = torch.zeros((X_np.shape[1], 1), dtype=torch.float32)
                    else:
                        clf = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=ConvergenceWarning)
                            clf.fit(X_np, y_np)
                        w_pred = torch.from_numpy(clf.coef_).float().unsqueeze(1)

                    test_x = xs[j, i : i + 1].float()
                    y_pred = (test_x @ w_pred).squeeze(1)
                    pred[j] = y_pred[0]
            preds.append(pred)
        return torch.stack(preds, dim=1)


class L1MinimizationModel:
    def __init__(self, epsilon=1e-6, solver="SCS"):
        self.epsilon = epsilon
        self.solver = solver
        self.name = f"l1_minimization_epsilon={epsilon}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])

        preds = []
        for i in inds:
            pred = torch.zeros_like(ys[:, 0], dtype=torch.float32)
            if i > 0:
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # Convert to numpy float32
                    X_np = train_xs.numpy().astype("float32")
                    y_np = train_ys.numpy().astype("float32")

                    # CVXPY variable
                    a = cp.Variable(X_np.shape[1])
                    objective = cp.Minimize(cp.norm(a, 1))
                    constraints = [cp.norm(X_np @ a - y_np, 2) <= self.epsilon]
                    problem = cp.Problem(objective, constraints)
                    try:
                        problem.solve(solver=self.solver, verbose=False)
                    except Exception as e:
                        print(f"[⚠️] CVXPY solver failed at i={i}, j={j}: {e}")
                        a.value = None

                    if a.value is None:
                        w_pred = torch.zeros((X_np.shape[1], 1), dtype=torch.float32)
                    else:
                        w_pred = torch.from_numpy(a.value.astype("float32")).unsqueeze(1)

                    test_x = xs[j, i : i + 1].float()
                    y_pred = (test_x @ w_pred).squeeze(1)
                    pred[j] = y_pred[0]
            preds.append(pred)
        return torch.stack(preds, dim=1)


class NuclearNormMinimizationModel:
    """
    Nuclear norm minimization.
    - Inputs (xs, ys) are torch tensors as in your other models.
    - Output: same format as LeastSquaresModel -> torch.stack(preds, dim=1)
    """

    def __init__(self, epsilon=1e-6, solver="SCS", reg=0):
        self.epsilon = float(epsilon)
        self.solver = solver
        self.reg = reg
        self.name = f"nuclear_norm_minimization_epsilon={self.epsilon}"

    def __call__(self, xs, ys, inds=None):
        """
        xs: torch.Tensor (batch, n_points, n_dim) where n_dim = n1 + n2
        ys: torch.Tensor (batch, n_points)
        inds: iterable of indices to predict (or None -> all)
        returns: torch.Tensor (batch, len(inds))
        """
        xs, ys = xs.cpu(), ys.cpu()
        batch_size, n_points, n_dim = xs.shape
        n1 = n2 = n_dim // 2

        if inds is None:
            inds = range(n_points)
        else:
            # ensure inds is list-like
            inds = list(inds)

        preds_per_timestep = []  # will collect tensors of shape (batch,) for each timestep

        for i in inds:
            if i == 0:
                # first prediction: zeros (same behavior as other models)
                preds_per_timestep.append(torch.zeros(batch_size, dtype=torch.float32))
                continue

            # For timestep i we produce one scalar per batch element
            batch_preds = torch.zeros(batch_size, dtype=torch.float32)

            for b in range(batch_size):
                # training data up to i for sample b
                X1_train = xs[b, :i, :n1].numpy().astype("float32")  # shape (i, n1)
                X2_train = xs[b, :i, n1:].numpy().astype("float32")  # shape (i, n2)
                y_train = ys[b, :i].numpy().astype("float32").reshape(-1, 1)  # shape (i,1)

                # If too few training points -> fallback to zero prediction
                if X1_train.shape[0] == 0:
                    batch_preds[b] = 0.0
                    continue

                # CVXPY variable
                A_var = cp.Variable((n1, n2))

                # Build CVXPY expression F(A) -> (i,1)
                F_expr = F_cvxpy(A_var, X1=X1_train, X2=X2_train)

                # Ensure y is a CVXPY constant shaped (i,1)
                y_const = cp.Constant(y_train.reshape(-1, 1))

                # Constraint: ||F(A)-y||_2 <= epsilon
                constraints = [cp.norm(F_expr - y_const, 2) <= self.epsilon]

                # Solve nuclear norm minimization
                prob = cp.Problem(cp.Minimize(cp.normNuc(A_var)), constraints)
                try:
                    prob.solve(solver=self.solver, verbose=False)
                except Exception as e:
                    # solver failed -> fallback to zeros matrix and warn once
                    # keep A_var.value = None
                    A_val = None

                A_val = A_var.value if (hasattr(A_var, "value") and A_var.value is not None) else None

                if A_val is None:
                    # solver failed -> predict 0
                    batch_preds[b] = 0.0
                else:
                    # compute prediction for the i-th test point (x1_test, x2_test)
                    x1_test = xs[b, i, :n1].numpy().astype("float32").reshape(1, n1)  # (1, n1)
                    x2_test = xs[b, i, n1:].numpy().astype("float32").reshape(1, n2)  # (1, n2)
                    # numeric F_cvxpy returns numpy array shape (1,)
                    pred_val = F_cvxpy(A_val.astype("float32"), X1=x1_test, X2=x2_test)
                    # pred_val can be array-like length 1 -> take [0]
                    batch_preds[b] = float(np.asarray(pred_val).reshape(-1)[0])

            preds_per_timestep.append(batch_preds)

        # Stack per-timestep predictions into (batch, len(inds))
        return torch.stack(preds_per_timestep, dim=1)
