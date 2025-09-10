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

    if xs_p is None:
        # Supprimer toute dimension batch supplémentaire
        if xs.dim() == 4:
            # xs: [batch, extra_batch, points, dim] -> [batch, points, dim]
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
    Tau = 0 → xs totalement aléatoires, train/test identiques
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    # Découpage train/test : moitié-moitié
    split = n_points // 2
    xs_train_pre = xs[:, :split, :]
    xs_test_post = xs[:, split:, :]
    print("In coherence tau=0 generation")
    return xs_train_pre, xs_test_post


def gen_coherence_tau_05(data_sampler, n_points, b_size):
    """
    Tau = 0.5 → moitié cohérence avec la base Phi, moitié bruit
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    Phi = torch.tensor(data_sampler.Phi, dtype=xs.dtype, device=xs.device)
    U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
    proj = U @ U.T

    # Appliquer la cohérence tau=0.5
    xs_coherent = 0.5 * (xs @ proj) + 0.5 * torch.randn_like(xs)

    # Découpage train/test
    split = n_points // 2
    xs_train_pre = xs_coherent[:, :split, :]
    xs_test_post = xs_coherent[:, split:, :]
    print("In coherence tau=0.5 generation")
    return xs_train_pre, xs_test_post


def gen_coherence_tau_1(data_sampler, n_points, b_size):
    """
    Tau = 1 → xs entièrement projetés sur la base Phi
    """
    xs = data_sampler.sample_xs(n_points, b_size)
    Phi = torch.tensor(data_sampler.Phi, dtype=xs.dtype, device=xs.device)
    U, _, _ = torch.linalg.svd(Phi, full_matrices=False)
    proj = U @ U.T

    xs_projected = xs @ proj

    # Découpage train/test
    split = n_points // 2
    xs_train_pre = xs_projected[:, :split, :]
    xs_test_post = xs_projected[:, split:, :]
    print("In coherence tau=1 generation")
    return xs_train_pre, xs_test_post




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

    # --- Cas standard ---
    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}

    # --- Cas Sparse Recovery (Compressed Sensing) ---
    if task_name == "compressed_sensing":
        # Trois versions de gen_coherence_tau() avec tau 0, 0.5 et 1
        evaluation_kwargs["tau=0"] = {"prompting_strategy": "coherence_tau_0"}
        evaluation_kwargs["tau=0.5"] = {"prompting_strategy": "coherence_tau_05"}
        evaluation_kwargs["tau=1"] = {"prompting_strategy": "coherence_tau_1"}

        # Exemple d'évaluation bruitée pour Sparse Recovery
        #evaluation_kwargs["noisy_sparse"] = {
        #    "prompting_strategy": "standard",  # ou "noisy" si tu as une fonction correspondante
        #    "task_sampler_kwargs": {"noise_std": 0.1},
        #}

    # --- Cas Matrix Factorization ---
    elif task_name == "matrix_factorization":
        for rank in [2, 5, 10]:
            evaluation_kwargs[f"rank={rank}"] = {
                "prompting_strategy": "standard",
                "task_sampler_kwargs": {"target_rank": rank},
            }
        #evaluation_kwargs["noisyMF"] = {
        #    "prompting_strategy": "standard",
        #    "task_sampler_kwargs": {"noise_std": 0.1},
        #}

    # --- Fusion finale avec base_kwargs ---
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
       - task: which base task we are evaluating on. E.g., "sparse recovery"
       - prompting_strategy: how to construct the prompt, e.g., "coherence_tau = 0"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

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
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        if not os.path.isdir(task_dir):  # ignorer les fichiers comme .DS_Store
            continue
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            if not os.path.isdir(run_path):  # ignorer fichiers non-dossiers
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


if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)