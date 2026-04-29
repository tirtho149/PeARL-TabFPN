"""PEaRL Survival Analysis: Cox PH Model & C-index"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


def compute_risk_scores(pathway_pred: np.ndarray) -> np.ndarray:
    """
    Compute risk scores from predicted pathway expression.

    Simple approach: weighted sum of pathways associated with survival.
    In practice, this would use pathway-to-survival associations.
    """
    # Placeholder: use mean pathway expression as risk score
    risk_scores = np.mean(pathway_pred, axis=1)
    return risk_scores


def simulate_survival_data(
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate censored survival data.

    Returns:
        times: survival times
        events: event indicators (0=censored, 1=event)
        durations: times or censoring times
    """
    np.random.seed(seed)

    # Exponential survival times
    times = np.random.exponential(scale=5.0, size=n_samples)

    # Random censoring times
    censoring_times = np.random.exponential(scale=7.0, size=n_samples)

    # Observed durations
    durations = np.minimum(times, censoring_times)

    # Event indicators
    events = (times <= censoring_times).astype(int)

    return times, events, durations


def compute_concordance_index(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute concordance index (C-index) for survival prediction.

    Args:
        risk_scores: predicted risk scores
        event_times: observed event/censoring times
        event_indicators: 1 if event, 0 if censored

    Returns:
        c_index: concordance index [0, 1]
        std: standard deviation (bootstrap estimated)
    """
    # Compute C-index
    c_idx = concordance_index(event_times, -risk_scores, event_indicators)

    # Bootstrap CI estimation
    n_bootstrap = 100
    bootstrap_cis = []
    np.random.seed(42)

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(risk_scores), size=len(risk_scores), replace=True)
        boot_ci = concordance_index(
            event_times[idx], -risk_scores[idx], event_indicators[idx]
        )
        bootstrap_cis.append(boot_ci)

    std = np.std(bootstrap_cis)

    return c_idx, std


def fit_cox_model(
    features: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    feature_names: list = None,
) -> Tuple[CoxPHFitter, Dict]:
    """
    Fit Cox Proportional Hazards model.

    Args:
        features: (n_samples, n_features) feature matrix
        event_times: observed times
        event_indicators: event indicators

    Returns:
        model: fitted CoxPHFitter
        results: summary dict with coefficients, p-values, concordance
    """
    # Prepare data
    n_samples, n_features = features.shape

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    data = pd.DataFrame(features, columns=feature_names)
    data["T"] = event_times
    data["E"] = event_indicators

    # Fit model
    cph = CoxPHFitter()
    cph.fit(data, duration_col="T", event_col="E")

    # Extract results
    summary = cph.summary
    results = {
        "concordance_index": cph.concordance_index_,
        "log_likelihood": cph.log_likelihood_,
        "coefficients": summary["coef"].to_dict(),
        "p_values": summary["p"].to_dict(),
    }

    return cph, results


def plot_survival_curves(
    risk_scores: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
    output_path: str = None,
):
    """Plot Kaplan-Meier survival curves stratified by risk score."""
    try:
        import matplotlib.pyplot as plt
        from lifelines import KaplanMeierFitter

        # Stratify by median risk score
        median_risk = np.median(risk_scores)
        high_risk = risk_scores > median_risk
        low_risk = ~high_risk

        kmf = KaplanMeierFitter()

        fig, ax = plt.subplots(figsize=(8, 6))

        # High risk
        kmf.fit(
            event_times[high_risk],
            event_indicators[high_risk],
            label=f"High Risk (n={np.sum(high_risk)})",
        )
        kmf.plot_survival_function(ax=ax, linewidth=2)

        # Low risk
        kmf.fit(
            event_times[low_risk], event_indicators[low_risk], label=f"Low Risk (n={np.sum(low_risk)})"
        )
        kmf.plot_survival_function(ax=ax, linewidth=2)

        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier Curves by Risk Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    except ImportError:
        print("lifelines not available for survival plotting")


def evaluate_survival_prediction(
    pathway_pred: np.ndarray,
    event_times: np.ndarray,
    event_indicators: np.ndarray,
) -> Dict[str, float]:
    """Full survival analysis pipeline."""
    # Compute risk scores
    risk_scores = compute_risk_scores(pathway_pred)

    # C-index
    c_idx, c_idx_std = compute_concordance_index(risk_scores, event_times, event_indicators)

    # Cox model
    cph, cox_results = fit_cox_model(pathway_pred, event_times, event_indicators)

    results = {
        "c_index": c_idx,
        "c_index_std": c_idx_std,
        "log_likelihood": cox_results["log_likelihood"],
    }

    return results
