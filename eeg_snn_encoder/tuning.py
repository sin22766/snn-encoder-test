import os
from typing import List, TypedDict

import optuna


class TrialFilter(TypedDict):
    """
    A dictionary to define the filter function for trials.
    
    Attributes:
        name (str): The name of the study.
        filter_fn (callable): A function that takes a trial and returns True if it should be included.
    """
    name: str
    filter_fn: callable


def filter_and_sort_trials(study_configs: List[TrialFilter], storage_url=None):
    """
    Load, filter, and sort trials from multiple Optuna studies based on custom filters.
    
    Args:
        study_configs (list): List of dictionaries containing study configurations.
                              Each dict should have 'name' and 'filter_fn' keys.
        storage_url (str, optional): The Optuna storage URL. If None, uses os.environ["OPTUNA_CONN_STRING"].
    
    Returns:
        list: Sorted filtered trials from all studies
    """
    
    # Use provided storage URL or fall back to environment variable
    storage = storage_url or os.environ.get("OPTUNA_CONN_STRING")
    if not storage:
        raise ValueError("No storage URL provided and OPTUNA_CONN_STRING environment variable not set")
    
    qualifying_trials = []
    
    for config in study_configs:
        study_name = config["name"]
        filter_fn = config["filter_fn"]
        
        print(f"Loading trials from {study_name}")
        current_study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )

        completed_trials = current_study.get_trials(
            False, states=(optuna.trial.TrialState.COMPLETE,)
        )
        qualifying_study_trials = list(filter(filter_fn, completed_trials))
        ranked_trials = sorted(qualifying_study_trials, key=lambda t: t.value)
        
        qualifying_trials.extend(ranked_trials)
        print(f"Found {len(ranked_trials)} qualifying trials in {study_name}")
    
    return qualifying_trials