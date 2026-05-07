import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance, title="Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#028090"] * len(importance)
    bars = ax.barh(importance.index, importance.values, color=colors, edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#028090", edgecolors="white", linewidth=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Happiness Score")
    ax.set_ylabel("Predicted Happiness Score")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

def plot_happiness_trend(df, countries):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#028090", "#02C39A", "#E63946", "#F4A261", "#A8DADC"]
    for i, country in enumerate(countries):
        subset = df[df["country"] == country].sort_values("year")
        ax.plot(subset["year"], subset["happiness_score"],
                marker="o", label=country, color=colors[i % len(colors)], linewidth=2)
    ax.set_title("Happiness Score Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Happiness Score")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig

def plot_model_comparison(results):
    models = list(results.keys())
    r2 = [results[m]["r2"] for m in models]
    mae = [results[m]["mae"] for m in models]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - width/2, r2, width, label="R²", color=["#028090", "#02C39A"])
    b2 = ax.bar(x + width/2, mae, width, label="MAE", color=["#A8DADC", "#E63946"])
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Baseline vs Improved Model", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    return fig
