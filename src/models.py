import torch
import torch.nn as nn
import cvxpy as cp

from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings


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
        "linear_regression": [
            (LeastSquaresModel, {}),
        ],
        "compressed_sensing": [
            (LeastSquaresModel, {}),
            (L1MinimizationModel, {"epsilon": 1e-6})
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001]],
        "matrix_factorization": [
            (LeastSquaresModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001]],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
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
        self._read_in = nn.Linear(n_dims, n_embd)
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

        if inds is None:
            inds = torch.arange(xs.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= xs.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)

        return prediction[:, ::2, 0][:, inds]  # predict only on xs




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


