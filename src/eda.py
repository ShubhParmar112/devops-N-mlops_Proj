"""Exploratory data analysis: prints a summary and saves plots to reports/figures/.

Usage:  python -m src.eda
"""
import matplotlib

matplotlib.use("Agg")  # render to files, no GUI window needed
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import FEATURES, PROJECT_ROOT, TARGET, ZERO_AS_MISSING, load_data

FIG_DIR = PROJECT_ROOT / "reports" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_data()

    print("=== Shape ===")
    print(df.shape)
    print("\n=== Summary statistics ===")
    print(df.describe().round(2).T)
    print("\n=== Null values (explicit NaN) ===")
    print(df.isnull().sum().sum(), "nulls")
    print("\n=== Zero values that actually mean 'missing' ===")
    zeros = (df[ZERO_AS_MISSING] == 0).sum()
    print(zeros.to_string())
    print("\n=== Class balance (Outcome) ===")
    print(df[TARGET].value_counts().rename({0: "Negative (0)", 1: "Positive (1)"}).to_string())
    print(f"Positive rate: {df[TARGET].mean():.1%}")

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, hue=TARGET, palette=["#4C9F70", "#D1495B"], legend=False, ax=ax)
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=120)
    plt.close(fig)

    # 2. Zero / missing counts
    fig, ax = plt.subplots(figsize=(6, 4))
    zeros.sort_values().plot.barh(ax=ax, color="#2E86AB")
    ax.set_title("Zero values treated as missing")
    ax.set_xlabel("Number of rows")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "zero_values.png", dpi=120)
    plt.close(fig)

    # 3. Feature distributions by outcome
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax,
                     palette=["#4C9F70", "#D1495B"], element="step")
        ax.set_title(col)
    fig.suptitle("Feature distributions by outcome (0 = negative, 1 = positive)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=120)
    plt.close(fig)

    # 4. Correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=120)
    plt.close(fig)

    # 5. Box plots by outcome (outliers)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.boxplot(data=df, x=TARGET, y=col, hue=TARGET, ax=ax,
                    palette=["#4C9F70", "#D1495B"], legend=False)
        ax.set_title(col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "boxplots_by_outcome.png", dpi=120)
    plt.close(fig)

    print(f"\nCorrelation with Outcome:\n{df.corr()[TARGET].drop(TARGET).sort_values(ascending=False).round(3).to_string()}")
    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
