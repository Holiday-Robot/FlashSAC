import re
from typing import Any, Optional, Sequence, Union

import torch

from flash_rl.agents.utils.network import Network
from flash_rl.agents.utils.tree import tree_filter, tree_leaves, tree_map

# Additional typings
Params = dict[str, Any]  # PyTorch uses regular dict instead of FrozenDict
Data = Union[torch.Tensor, dict[str, "Data"]]
Batch = dict[str, Data]


# rephrase each key
def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = "_") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def add_prefix_to_dict(d: dict[str, Any], prefix: str = "", sep: str = "/") -> dict[str, Any]:
    new_dict: dict[str, Any] = {}
    for key, value in d.items():
        new_dict[prefix + sep + key] = value
    return new_dict


def sum_all_values_in_pytree(pytree: dict[str, Any]) -> torch.Tensor:
    # Flatten the pytree to get all leaves (individual values)
    leaves = tree_leaves(pytree)
    # Sum all leaves
    total_sum = sum(torch.sum(leaf) for leaf in leaves if isinstance(leaf, torch.Tensor))
    return torch.tensor(float(total_sum), dtype=torch.float32)


def get_pnorm(
    param_dict: Params,
    pcount_dict: Params,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    param_dict is a dictionary which contains the values of each individual parameter

    Return:
        param value norm dictionary
        (CAUTION : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    _pnorm_dict = tree_map(lambda x: torch.norm(x), param_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_pnorm = add_all_key(_pnorm_dict)

    # Construct a pnorm dict
    pnorm_dict = {}
    eff_pnorm_numer, eff_pnorm_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_pnorm_total_numer, eff_pnorm_total_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_pnorm_encoder_numer, eff_pnorm_encoder_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_pnorm_predictor_numer, eff_pnorm_predictor_denom = torch.tensor(0.0), torch.tensor(0.0)
    pnorm_total = torch.tensor(0.0)
    pnorm_encoder_total = torch.tensor(0.0)
    pnorm_predictor_total = torch.tensor(0.0)
    for module, layer_dict in _updated_pnorm.items():
        # e.g. module = "encoder" or "predictor"
        pnorm_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                prefix + "_" + module + "/pnorm",
                sep="_",
            )
        )

    # Aggregation
    for _pnorm_layer, _pnorm in pnorm_dict.items():
        # e.g. _module = actor_encoder, _layer_name = pnorm_Dense_0_bias
        _module, _layer_name = _pnorm_layer.split("/", 1)
        _layer = _layer_name.replace("pnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer):
            continue
        pnorm_total += torch.square(_pnorm)

        # Compute effective parameter norms
        eff_pnorm_total_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_pnorm)
        eff_pnorm_total_denom += float(pcount_dict[_module + "/pcount_" + _layer])

        eff_pnorm_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_pnorm)
        eff_pnorm_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        if "encoder" in _module:
            pnorm_encoder_total += torch.square(_pnorm)
            eff_pnorm_encoder_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_pnorm)
            eff_pnorm_encoder_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        elif "predictor" in _module:
            pnorm_predictor_total += torch.square(_pnorm)
            eff_pnorm_predictor_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_pnorm)
            eff_pnorm_predictor_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        else:
            raise NotImplementedError

    pnorm_dict[prefix + "/pnorm_total"] = torch.sqrt(pnorm_total)
    pnorm_dict[prefix + "_encoder/pnorm_total"] = torch.sqrt(pnorm_encoder_total)
    pnorm_dict[prefix + "_predictor/pnorm_total"] = torch.sqrt(pnorm_predictor_total)

    pnorm_dict[prefix + "/effective_pnorm_total"] = torch.sqrt(eff_pnorm_numer / eff_pnorm_denom)
    pnorm_dict[prefix + "_encoder/effective_pnorm_total"] = torch.sqrt(
        eff_pnorm_encoder_numer / eff_pnorm_encoder_denom
    )
    pnorm_dict[prefix + "_predictor/effective_pnorm_total"] = torch.sqrt(
        eff_pnorm_predictor_numer / eff_pnorm_predictor_denom
    )

    return pnorm_dict


def get_gnorm(
    grad_dict: Params,
    pcount_dict: Params,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    grad_dict is a dictionary which contains the gradients of each individual parameter

    Return:
        param gradient norm dictionary
        (CAUTION : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    _gnorm_dict = tree_map(lambda x: torch.norm(x) if isinstance(x, torch.Tensor) else x, grad_dict)
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_gnorm = add_all_key(_gnorm_dict)

    # Construct a gnorm dict
    gnorm_dict = {}
    eff_gnorm_numer, eff_gnorm_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_gnorm_total_numer, eff_gnorm_total_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_gnorm_encoder_numer, eff_gnorm_encoder_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_gnorm_predictor_numer, eff_gnorm_predictor_denom = torch.tensor(0.0), torch.tensor(0.0)
    gnorm_total = torch.tensor(0.0)
    gnorm_encoder_total = torch.tensor(0.0)
    gnorm_predictor_total = torch.tensor(0.0)
    for module, layer_dict in _updated_gnorm.items():
        # e.g. module = "encoder" or "predictor"
        gnorm_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                prefix + "_" + module + "/gnorm",
                sep="_",
            )
        )

    # Aggregation
    for _gnorm_layer, _gnorm in gnorm_dict.items():
        # e.g. _module = actor_encoder, _layer_name = gnorm_Dense_0_bias
        _module, _layer_name = _gnorm_layer.split("/", 1)
        _layer = _layer_name.replace("gnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer):
            continue
        gnorm_total += torch.square(_gnorm)

        # Compute effective parameter norms
        eff_gnorm_total_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_gnorm)
        eff_gnorm_total_denom += float(pcount_dict[_module + "/pcount_" + _layer])

        eff_gnorm_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_gnorm)
        eff_gnorm_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        if "encoder" in _module:
            gnorm_encoder_total += torch.square(_gnorm)
            eff_gnorm_encoder_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_gnorm)
            eff_gnorm_encoder_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        elif "predictor" in _module:
            gnorm_predictor_total += torch.square(_gnorm)
            eff_gnorm_predictor_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * torch.square(_gnorm)
            eff_gnorm_predictor_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        else:
            raise NotImplementedError

    gnorm_dict[prefix + "/gnorm_total"] = torch.sqrt(gnorm_total)
    gnorm_dict[prefix + "_encoder/gnorm_total"] = torch.sqrt(gnorm_encoder_total)
    gnorm_dict[prefix + "_predictor/gnorm_total"] = torch.sqrt(gnorm_predictor_total)

    gnorm_dict[prefix + "/effective_gnorm_total"] = torch.sqrt(eff_gnorm_numer / eff_gnorm_denom)
    gnorm_dict[prefix + "_encoder/effective_gnorm_total"] = torch.sqrt(
        eff_gnorm_encoder_numer / eff_gnorm_encoder_denom
    )
    gnorm_dict[prefix + "_predictor/effective_gnorm_total"] = torch.sqrt(
        eff_gnorm_predictor_numer / eff_gnorm_predictor_denom
    )

    return gnorm_dict


def get_effective_lr(
    gnorm_dict: Params,
    pnorm_dict: Params,
    pcount_dict: Params,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    grad_dict is a dictionary which contains the gradients of each individual parameter

    Return:
        param gradient norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    eff_lr_dict = {}
    eff_lr_encoder_numer, eff_lr_encoder_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_lr_predictor_numer, eff_lr_predictor_denom = torch.tensor(0.0), torch.tensor(0.0)
    eff_lr_total_numer, eff_lr_total_denom = torch.tensor(0.0), torch.tensor(0.0)

    for _gnorm_layer, _gnorm in gnorm_dict.items():
        # e.g. module = actor_encoder, _layer_name = gnorm_Dense_0_bias
        _module, _layer_name = _gnorm_layer.split("/", 1)
        _layer = _layer_name.replace("gnorm_", "")
        if ("kernel+bias" in _layer) or ("total" in _layer) or ("effective" in _layer):
            continue
        eff_lr = _gnorm / pnorm_dict[_module + "/pnorm_" + _layer]
        eff_lr_dict[_module + "/effective_lr_" + _layer] = eff_lr

        # Aggregation
        eff_lr_total_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * eff_lr
        eff_lr_total_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        if "encoder" in _module:
            eff_lr_encoder_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * eff_lr
            eff_lr_encoder_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        elif "predictor" in _module:
            eff_lr_predictor_numer += float(pcount_dict[_module + "/pcount_" + _layer]) * eff_lr
            eff_lr_predictor_denom += float(pcount_dict[_module + "/pcount_" + _layer])
        else:
            raise NotImplementedError

    eff_lr_dict[prefix + "_encoder/effective_lr_total"] = eff_lr_encoder_numer / eff_lr_encoder_denom
    eff_lr_dict[prefix + "_predictor/effective_lr_total"] = eff_lr_predictor_numer / eff_lr_predictor_denom
    eff_lr_dict[prefix + "/effective_lr_total"] = eff_lr_total_numer / eff_lr_total_denom

    return eff_lr_dict


def get_scaler_statistics(
    param_dict: Params,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """
    param_dict is a dictionary which contains the gradients/values of each individual parameter

    Return:
        param gradient/value norm dictionary
        (Caution : norm values for vmapped functions (multi-head Q-networks) are summed to a single value)
    """
    regex = "scaler"
    mean = tree_filter(
        f=lambda x: torch.mean(x) if isinstance(x, torch.Tensor) else x,
        tree=param_dict,
        target_re=regex,
    )
    var = tree_filter(
        f=lambda x: torch.var(x) if isinstance(x, torch.Tensor) else x,
        tree=param_dict,
        target_re=regex,
    )

    assert isinstance(mean, dict)
    assert isinstance(var, dict)
    modules = list(set(mean.keys()))  # e.g., modules = ["encoder", "predictor"]
    scaler_mean_dict, scaler_var_dict = {}, {}
    for module in modules:
        mean_layer_dict = mean[module]
        var_layer_dict = var[module]

        _mean_layer_dict = flatten_dict(mean_layer_dict)
        for _k, _v in _mean_layer_dict.items():
            # remove redundant key in `Scale``
            _k = _k.replace("_scaler", "")
            scaler_mean_dict[prefix + "_" + module + "/scaler-mean_" + _k] = _v

        _var_layer_dict = flatten_dict(var_layer_dict)
        for _k, _v in _var_layer_dict.items():
            # remove redundant key in `Scale``
            _k = _k.replace("_scaler", "")
            scaler_var_dict[prefix + "_" + module + "/scaler-var_" + _k] = _v

    info = {}
    info.update(scaler_mean_dict)
    info.update(scaler_var_dict)

    return info


def add_all_key(d: dict[str, Any]) -> dict[str, Any]:
    new_dict: dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            new_dict[key] = add_all_key(value)
            if "kernel" in new_dict[key] and "bias" in new_dict[key]:
                kernel_norm = torch.square(new_dict[key]["kernel"])
                bias_norm = torch.square(new_dict[key]["bias"])
                # Integrated Norm
                new_dict[key + "_kernel+bias"] = torch.sqrt(kernel_norm + bias_norm)
                # Separated Norm
                new_dict[key + "_kernel"] = torch.sqrt(kernel_norm)
                new_dict[key + "_bias"] = torch.sqrt(bias_norm)
        else:
            new_dict[key] = torch.norm(value) if isinstance(value, torch.Tensor) else value
    return new_dict


def get_dormant_ratio(
    activations: dict[str, list[torch.Tensor]], prefix: str, tau: float = 0.1
) -> dict[str, torch.Tensor]:
    """
    Compute the dormant mask for a given set of activations.

    Args:
        activations: A dictionary of activations.
        prefix: A string prefix for naming.
        tau: The threshold for determining dormancy.

    Returns:
        A dictionary of dormancy ratios for each layer and the total.

    Source : https://github.com/timoklein/redo/blob/dcaeff1c6afd0f1615a21da5beda870487b2ed15/src/redo.py#L215
    """
    key = "dormant" if tau > 0.0 else "zeroactiv"
    ratios = {}
    total_activs = []

    for sub_layer_name, activs in list(activations.items()):
        module, layer_name = sub_layer_name.split("/", 1)

        # Convert list of arrays to a single array (for double critics, stack them)
        activs_array: torch.Tensor
        if isinstance(activs, list):
            activs_array = torch.cat(activs, dim=0) if len(activs) > 0 else torch.tensor([])
        else:
            activs_array = activs

        # For double critics, lets just stack them into one batch
        if len(activs_array.shape) > 2:
            activs_array = activs_array.reshape(-1, activs_array.shape[-1])

        # Taking the mean here conforms to the expectation under D in the main paper's formula
        score = torch.abs(activs_array).mean(dim=0)
        # Divide by activation mean to make the threshold independent of the layer size
        # see https://github.com/google/dopamine/blob/ce36aab6528b26a699f5f1cefd330fdaf23a5d72/dopamine/labs/redo/weight_recyclers.py#L314
        # https://github.com/google/dopamine/issues/209
        normalized_score = score / (score.mean() + 1e-9)

        if tau > 0.0:
            layer_mask = torch.where(
                normalized_score <= tau, torch.ones_like(normalized_score), torch.zeros_like(normalized_score)
            )
        else:
            layer_mask = torch.where(
                torch.isclose(normalized_score, torch.zeros_like(normalized_score)),
                torch.ones_like(normalized_score),
                torch.zeros_like(normalized_score),
            )

        ratios[f"{prefix}_{module}/{key}_{layer_name}"] = (torch.sum(layer_mask) / layer_mask.numel()) * 100
        total_activs.append(layer_mask)

    # aggregated mask of entire network
    total_mask = torch.cat(total_activs)

    ratios[f"{prefix}_{module}/{key}_total"] = (torch.sum(total_mask) / total_mask.numel()) * 100

    return ratios


# source: https://github.com/CLAIRE-Labo/no-representation-no-trust/blob/52a785da4aee93b569d87a289b1f5865271aedfe/src/po_dynamics/modules/metrics.py#L9
def get_rank(activations: dict[str, list[torch.Tensor]], prefix: str, tau: float = 0.01) -> dict[str, torch.Tensor]:
    """
    Computes different approximations of the rank for a given set of activations.

    Args:
        activations: A dictionary of activations.
        prefix: A string prefix for naming.
        tau: cutoff parameter. not used in (1), 1 - 99% in (2), delta in (3), epsilon in (4).

    Returns:
        (1) Effective rank.
        A continuous approximation of the rank of a matrix.
        Definition 2.1. in Roy & Vetterli, (2007) https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7098875
        Also used in Huh et al. (2023) https://arxiv.org/pdf/2103.10427.pdf

        (2) Approximate rank.
        Threshold at the dimensions explaining 99% of the variance in a PCA analysis.
        Section 2 in Yang et al. (2020) https://arxiv.org/pdf/1909.12255.pdf

        (3) srank.
        Another version of (2).
        Section 3 in Kumar et al. https://arxiv.org/pdf/2010.14498.pdf

        (4) Feature rank.
        A threshold rank: normalize by dim size and discard dimensions with singular values below 0.01.
        Equations (4) and (5). Lyle et al. (2022) https://arxiv.org/pdf/2204.09560.pdf

        (5) Torch rank.
        Rank defined in torch. A reasonable value for cutoff parameter is chosen based the floating point precision of
        the input.
    """
    threshold = 1 - tau
    ranks = {}

    for sub_layer_name, feature in list(activations.items()):
        module, layer_name = sub_layer_name.split("/", 1)

        # Convert list of arrays to a single array (for double critics, stack them)
        feature_array: torch.Tensor
        if isinstance(feature, list):
            feature_array = torch.cat(feature, dim=0) if len(feature) > 0 else torch.tensor([])
        else:
            feature_array = feature

        # For double critics, lets just stack them into one batch
        if len(feature_array.shape) > 2:
            feature_array = feature_array.reshape(-1, feature_array.shape[-1])

        # Compute the L2 norm for all examples in the batch at once
        svals = torch.linalg.svdvals(feature_array)

        # (1) Effective rank.
        sval_sum = torch.sum(svals)
        sval_dist = svals / sval_sum
        # Replace 0 with 1. This is a safe trick to avoid log(0) = -inf
        # as Roy & Vetterli assume 0*log(0) = 0 = 1*log(1).
        sval_dist_fixed = torch.where(sval_dist == 0, torch.ones_like(sval_dist), sval_dist)
        effective_ranks = torch.exp(-torch.sum(sval_dist_fixed * torch.log(sval_dist_fixed)))

        # (2) Approximate rank. PCA variance. Yang et al. (2020)
        sval_squares = svals**2
        sval_squares_sum = torch.sum(sval_squares)
        cumsum_squares = torch.cumsum(sval_squares, dim=0)
        threshold_crossed = cumsum_squares >= (threshold * sval_squares_sum)
        approximate_ranks = (~threshold_crossed).sum() + 1

        # (3) srank. Weird. Kumar et al. (2020)
        cumsum = torch.cumsum(svals, dim=0)
        threshold_crossed = cumsum >= threshold * sval_sum
        sranks = (~threshold_crossed).sum() + 1

        # (4) Feature rank. Most basic. Lyle et al. (2022)
        n_obs = feature_array.shape[0]
        svals_of_normalized = svals / torch.sqrt(torch.tensor(n_obs, dtype=svals.dtype))
        over_cutoff = svals_of_normalized > tau
        feature_ranks = over_cutoff.sum()

        # (5) torch rank.
        # Note that this determines the matrix rank same with (4), but some reasonable tau is chosen automatically
        # based on the floating point precision of the input.
        torch_ranks = torch.linalg.matrix_rank(feature_array)

        ranks.update(
            {
                f"{prefix}_{module}/effective-rank-vetterli_{layer_name}": effective_ranks,
                f"{prefix}_{module}/approximate-rank-pca_{layer_name}": approximate_ranks,
                f"{prefix}_{module}/srank-kumar_{layer_name}": sranks,
                f"{prefix}_{module}/feature-rank-lyle_{layer_name}": feature_ranks,
                f"{prefix}_{module}/matrix-rank_{layer_name}": torch_ranks,
            }
        )

    return ranks


def get_feature_norm(activations: dict[str, list[torch.Tensor]], prefix: str) -> dict[str, torch.Tensor]:
    """
    Computes the feature norm for a given set of activations.
    """
    norms = {}
    total_norm = torch.tensor(0.0)
    module_to_total_norm: dict[str, torch.Tensor] = {}
    for sub_layer_name, activs in list(activations.items()):
        # e.g. sub_layer_name = "encoder/Dense_0"
        module, layer_name = sub_layer_name.split("/", 1)

        # Convert list of arrays to a single array (for double critics, stack them)
        activs_array: torch.Tensor
        if isinstance(activs, list):
            activs_array = torch.cat(activs, dim=0) if len(activs) > 0 else torch.tensor([])
        else:
            activs_array = activs

        # For double critics, lets just stack them into one batch
        if len(activs_array.shape) > 2:
            activs_array = activs_array.reshape(-1, activs_array.shape[-1])

        # Compute the L2 norm for all examples in the batch at once
        batch_norms = torch.norm(activs_array, p=2, dim=-1)

        # Compute the expected (mean) L2 norm across the batch
        expected_norm = torch.mean(batch_norms)

        norms[f"{prefix}_{module}/featnorm_{layer_name}"] = expected_norm
        total_norm += expected_norm

        module_to_total_norm[module] = module_to_total_norm.get(module, torch.tensor(0.0)) + torch.sum(
            torch.square(activs_array)
        )

    norms[f"{prefix}/featnorm_total"] = total_norm
    for module in module_to_total_norm.keys():
        norms[f"{prefix}_{module}/featnorm_total"] = torch.sqrt(module_to_total_norm[module])

    return norms


# source : https://arxiv.org/pdf/2112.04716
def get_critic_featdot(actor: Network, critic: Network, batch: Batch, sample: bool = True) -> dict[str, torch.Tensor]:
    if sample:
        dist, _ = actor(observations=batch["observation"])
        cur_actions = dist.sample()

        next_dist, _ = actor(observations=batch["next_observation"])
        next_actions = next_dist.sample()
    else:
        cur_actions, _ = actor(observations=batch["observation"])
        next_actions, _ = actor(observations=batch["next_observation"])

    _, cur_critic_info = critic(observations=batch["observation"], actions=cur_actions)

    final_cur_critic_feat = cur_critic_info[get_last_layer(cur_critic_info)]
    if len(final_cur_critic_feat.shape) > 2:
        final_cur_critic_feat = final_cur_critic_feat.reshape(-1, final_cur_critic_feat.shape[-1])

    _, next_critic_info = critic(observations=batch["next_observation"], actions=next_actions)

    final_next_critic_feat = next_critic_info[get_last_layer(next_critic_info)]
    if len(final_next_critic_feat.shape) > 2:
        final_next_critic_feat = final_next_critic_feat.reshape(-1, final_next_critic_feat.shape[-1])

    # Compute mean dot product of the batch
    # Don't do cosine similarity, it has to be dot product according to the paper
    result = torch.mean(torch.sum(final_cur_critic_feat * final_next_critic_feat, dim=1), dim=0)

    return {"critic/DR3_featdot": result}


def get_last_layer(layer_dict: dict[str, Any]) -> Optional[str]:
    def extract_number(key: str) -> int:
        match = re.search(r"\d+$", key)
        return int(match.group()) if match else -1

    # Sort keys based on the numeric suffix
    sorted_keys = sorted(layer_dict.keys(), key=extract_number, reverse=True)

    # Return the first key (which will be the one with the highest number)
    return sorted_keys[0] if sorted_keys else None


def print_num_parameters(pytree_dict_list: Sequence[dict[str, Any]], network_type: str) -> None:
    """
    Return number of trainable parameters
    """
    total_params = 0

    for pytree_dict in pytree_dict_list:
        leaf_nodes = tree_leaves(pytree_dict)
        for leaf in leaf_nodes:
            if isinstance(leaf, torch.Tensor):
                total_params += int(torch.prod(torch.tensor(leaf.shape)))

    # Format the total_params to a human-readable string
    if total_params >= 1e6:
        print(f"{network_type} total params: {total_params / 1e6:.2f}M")
    elif total_params >= 1e3:
        print(f"{network_type} total params: {total_params / 1e3:.2f}K")
    else:
        print(f"{network_type} total params: {total_params}")


def get_num_parameters_dict(
    param_dict: Params,
    prefix: str,
) -> dict[str, int]:
    """
    Return dictionary that contains number of trainable parameters for each layer
    """
    _pcount_dict = tree_map(
        lambda x: int(torch.prod(torch.tensor(x.shape))) if isinstance(x, torch.Tensor) else x, param_dict
    )
    # Add 'kernel', 'bias', and 'kernel+bias' norm for each layer
    _updated_pcount = add_all_key(_pcount_dict)

    # Construct a pcount dict
    pcount_dict = {}
    for module, layer_dict in _updated_pcount.items():
        # e.g. module = "encoder" or "predictor"
        pcount_dict.update(
            add_prefix_to_dict(
                flatten_dict(layer_dict),
                prefix + "_" + module + "/pcount",
                sep="_",
            )
        )

    return pcount_dict
