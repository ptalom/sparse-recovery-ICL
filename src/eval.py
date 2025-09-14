import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf
    
    # --- Ajustement spécifique Matrix Factorization ---
    if conf.training.task == "matrix_factorization":
        conf.model.n_dims = conf.training.task_kwargs.n1 + conf.training.task_kwargs.n2
        conf.training.task_kwargs.n1 = conf.training.task_kwargs.n1
        conf.training.task_kwargs.n2 = conf.training.task_kwargs.n2

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    device = "cuda" if (torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]) else "cpu"

    # --- Cas Matrix Factorization ---
    if task.__class__.__name__ == "MatrixFactorizationTask": 
        ys = task.evaluate(xs)
        pred = model(xs.to(device), ys.to(device)).detach()
        metrics = task.get_metric()(pred.cpu(), ys)

    # --- Cas standard pour les autres tâches ---
    else:
        if xs_p is None:
            if xs.dim() == 4:
                xs = xs[:, 0, :, :]
            ys = task.evaluate(xs)
            pred = model(xs.to(device), ys.to(device)).detach()
            metrics = task.get_metric()(pred.cpu(), ys)
        else:
            b_size, n_points, _ = xs.shape
            metrics = torch.zeros(b_size, n_points)
            for i in range(n_points):
                xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
                ys = task.evaluate(xs_comb)
                pred = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
                metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


#---------------------------------------------------
# -----------------------------
# Fonctions de génération pour Sparse Recovery
# -----------------------------

def gen_standard(data_sampler, n_points, b_size):
    xs_list = []
    for _ in range(b_size):
        xs = data_sampler.sample_xs(n_points, b_size)
        xs_list.append(xs.squeeze(0))
    xs_train = torch.stack(xs_list, dim=0)
    print("In standard generation")
    return xs_train, None



def gen_coherence_tau_0(data_sampler, n_points, b_size):
    """
    Tau = 0 → xs totalement aléatoires
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, b_size)
    split = n_points_total // 2
    xs_train = xs[:, :split, :]
    xs_test = xs[:, split:, :]
    print("In coherence tau=0 generation")
    return xs_train, xs_test


def gen_coherence_tau_05(data_sampler, n_points, b_size):
    """
    Tau = 0.5 → moitié cohérence avec la base Phi, moitié bruit
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, b_size)
    Phi = torch.tensor(data_sampler.Phi, dtype=xs.dtype, device=xs.device)
    U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
    proj = U @ U.T

    # Appliquer la cohérence tau=0.5
    xs_coh = 0.5 * (xs @ proj) + 0.5 * torch.randn_like(xs)

    # Découpage train/test
    split = n_points_total // 2
    xs_train = xs_coh[:, :split, :]
    xs_test = xs_coh[:, split:, :]
    print("In coherence tau=0.5 generation")
    return xs_train, xs_test


def gen_coherence_tau_1(data_sampler, n_points, b_size):
    """
    Tau = 1 → xs entièrement projetés sur la base Phi
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, b_size)
    Phi = torch.tensor(data_sampler.Phi, dtype=xs.dtype, device=xs.device)
    U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
    proj = U @ U.T

    xs_projected = xs @ proj

    # Découpage train/test
    split = n_points_total // 2
    xs_train = xs_projected[:, :split, :]
    xs_test = xs_projected[:, split:, :]
    print("In coherence tau=1 generation")
    return xs_train, xs_test

#-----------------------------------------
# -----------------------------
# Fonctions de génération pour Matrix Factorization
# -----------------------------

def gen_standard_mf(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, batch_size=b_size)
    X1, X2 = xs[..., :data_sampler.n1], xs[..., data_sampler.n1:]
    xs_train = torch.cat([X1, X2], dim=-1)

    print("In standard generation MF")
    return xs_train, None


def gen_coherence_tau_0_mf(data_sampler, n_points, b_size):
    """
    Tau = 0 → xs totalement aléatoires
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, batch_size=b_size)
    X1, X2 = xs[..., :data_sampler.n1], xs[..., data_sampler.n1:]
    xs_combined = torch.cat([X1, X2], dim=-1)

    split = n_points_total // 2
    xs_train, xs_test = xs_combined[:, :split, :], xs_combined[:, split:, :]
    print("In MF generation tau=0")
    return xs_train, xs_test



def gen_coherence_tau_05_mf(data_sampler, n_points, b_size):
    """
    Tau = 0.5 → moitié cohérence avec la structure low-rank (U*V*ᵀ), moitié bruit.
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, batch_size=b_size)

    # Split en X1/X2 selon n1
    X1, X2 = xs[..., :data_sampler.n1], xs[..., data_sampler.n1:]

    # Récupérer U* et V* du data_sampler
    U_star = torch.tensor(data_sampler.U_star, dtype=X1.dtype, device=X1.device)
    V_star = torch.tensor(data_sampler.V_star, dtype=X1.dtype, device=X1.device)

    # Projection partielle (50%) + bruit (50%)
    X1_coh = 0.5 * (X1 @ V_star @ U_star.T) + 0.5 * torch.randn_like(X1)
    X2_coh = 0.5 * (X2 @ V_star @ U_star.T) + 0.5 * torch.randn_like(X2)

    xs_combined = torch.cat([X1_coh, X2_coh], dim=-1)

    split = n_points_total // 2
    xs_train = xs_combined[:, :split, :]
    xs_test  = xs_combined[:, split:, :]

    print("In MF generation tau=0.5")
    return xs_train, xs_test


def gen_coherence_tau_1_mf(data_sampler, n_points, b_size):
    """
    Tau = 1 pour Matrix Factorization → xs entièrement cohérents avec la matrice de rang faible.
    On split les xs retournés par data_sampler en X1/X2 puis on projette via U_star et V_star.
    """
    n_points_total = n_points * 2
    xs = data_sampler.sample_xs(n_points_total, batch_size=b_size) 

    X1, X2 = xs[..., :data_sampler.n1], xs[..., data_sampler.n1:]

    # Récupérer U_star et V_star du data_sampler
    U_star = torch.tensor(data_sampler.U_star, dtype=X1.dtype, device=X1.device)  # shape: (n1, rank)
    V_star = torch.tensor(data_sampler.V_star, dtype=X1.dtype, device=X1.device)  # shape: (n2, rank)

    X1_projected = (X1 @ V_star) @ U_star.T
    X2_projected = (X2 @ V_star) @ U_star.T

    xs_combined = torch.cat([X1_projected, X2_projected], dim=-1) 

    split = n_points_total // 2
    xs_train = xs_combined[:, :split, :]
    xs_test  = xs_combined[:, split:, :]

    print("In MF generation tau=1")
    return xs_train, xs_test



def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    # --- Cas Sparse Recovery ---
    if task_name == "compressed_sensing":
        
        evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
        evaluation_kwargs["tau=0"] = {"prompting_strategy": "coherence_tau_0"}
        evaluation_kwargs["tau=0.5"] = {"prompting_strategy": "coherence_tau_05"}
        evaluation_kwargs["tau=1"] = {"prompting_strategy": "coherence_tau_1"}

        # Evaluation bruitée pour Sparse Recovery
        #evaluation_kwargs["noisy_sparse"] = {
        #    "prompting_strategy": "standard", 
        #    "task_sampler_kwargs": {"noise_std": 0.1},
        #}

    # --- Cas Matrix Factorization ---
    elif task_name == "matrix_factorization":

        evaluation_kwargs["standard"] = {"prompting_strategy": "standard_mf"}
        for rank in [5, 10]:
            evaluation_kwargs[f"rank={rank}"] = {
                "prompting_strategy": "standard_mf",
                "task_sampler_kwargs": {"target_rank": rank},
            }

        evaluation_kwargs["tau=0"] = {"prompting_strategy": "coherence_tau_0_mf"}
        evaluation_kwargs["tau=0.5"] = {"prompting_strategy": "coherence_tau_05_mf"}
        evaluation_kwargs["tau=1"] = {"prompting_strategy": "coherence_tau_1_mf"}

        #evaluation_kwargs["noisyMF"] = {
        #    "prompting_strategy": "standard",
        #    "task_sampler_kwargs": {"noise_std": 0.1},
        #}

    for name, kwargs in evaluation_kwargs.items():
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}

def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task_name: which base task we are evaluating on. E.g., "sparse recovery"
       - data_name: which dataset/sampler to use
       - n_dims: input dimension
       - n_points: number of points in each sample
       - prompting_strategy: how to construct the prompt, e.g., "coherence_tau = 0"
       - num_eval_examples: total number of examples to evaluate on
       - batch_size: batch size
       - data_sampler_kwargs: arguments to pass to the data sampler
       - task_sampler_kwargs: arguments to pass to the task sampler
    """

    assert num_eval_examples % batch_size == 0

    # --- Ajustement pour Matrix Factorization ---
    if task_name == "matrix_factorization":
        # n_dims du modèle = n1 + n2
        n_dims = data_sampler_kwargs.get("n1", 20) + data_sampler_kwargs.get("n2", 20)

    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(task_name, n_dims, batch_size, **task_sampler_kwargs)

    # --- Synchronisation pour Matrix Factorization ---
    if task_name == "matrix_factorization":
        task_instance = task_sampler()
        if hasattr(task_instance, "update_from_sampler"):
            task_instance.update_from_sampler(data_sampler)
        else:
            task_instance.n1 = getattr(data_sampler, "n1", task_instance.n1)
            task_instance.n2 = getattr(data_sampler, "n2", task_instance.n2)
            task_instance.A_star = getattr(data_sampler, "A_star", getattr(task_instance, "A_star", None))
            
            if hasattr(task_instance, "A_star"):
                task_instance.A_b = torch.tensor(task_instance.A_star, dtype=torch.float32)

        # On fixe task_sampler pour renvoyer cette instance
        task_sampler = lambda: task_instance
        print(f"[DEBUG] MF task synchronisée avec data_sampler : n1={task_instance.n1}, n2={task_instance.n2}, A_star shape={getattr(task_instance,'A_star').shape}")
    
    # --- Boucle d'évaluation ---
    all_metrics = []
    generating_func = globals()[f"gen_{prompting_strategy}"]

    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)
        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)
    return aggregate_metrics(metrics)



def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics
    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics

def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model = model.to(device).eval()
        all_models = [model]
        if not skip_baselines:
            all_models += models.get_relevant_baselines(conf.training.task)
    
    # --- Ajustements spécifiques Matrix Factorization ---
    if conf.training.task == "matrix_factorization":
        conf.model.n_dims = conf.training.task_kwargs.n1 + conf.training.task_kwargs.n2
        conf.training.task_kwargs.n1 = conf.training.task_kwargs.n1
        conf.training.task_kwargs.n2 = conf.training.task_kwargs.n2

    
    evaluation_kwargs = build_evals(conf)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "l1_minimization" in name:
        epsilon = name.split("_")[-1].split("=")[1]
        return f"L1 Minimization (epsilon={epsilon})"
    if "nuclear_norm" in name:
        epsilon = name.split("_")[-1].split("=")[1]
        return f"Nuclear Norm Minimization (epsilon={epsilon})"
    if "gpt2" in name:
        return f"Transformer"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        if not os.path.isdir(task_dir): 
            continue
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            if not os.path.isdir(run_path):  
                continue

            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = getattr(conf.training, "num_tasks", None)
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = getattr(conf.training, "num_training_examples", None)
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df

'''
if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)

'''

if __name__ == "__main__":
    run_dir = sys.argv[1]

    def is_run_path(path):
        """Vérifie si le dossier contient un config.yaml → c'est un run UUID"""
        return os.path.isfile(os.path.join(path, "config.yaml"))

    def process_run_path(run_path):
        """Évalue un run spécifique"""
        print(f"Evaluating run {run_path}")
        metrics = get_run_metrics(run_path)
        return metrics

    if is_run_path(run_dir):
        process_run_path(run_dir)
    else:
        for task in os.listdir(run_dir):
            task_dir = os.path.join(run_dir, task)
            if not os.path.isdir(task_dir):
                continue
            print(f"Evaluating task {task}")
            for run_id in tqdm(os.listdir(task_dir)):
                run_path = os.path.join(task_dir, run_id)
                if is_run_path(run_path):
                    process_run_path(run_path)
