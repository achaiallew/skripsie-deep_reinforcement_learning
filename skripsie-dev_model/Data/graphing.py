import sys
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# SETTINGS - Edit these values, then run the script
# ============================================================

# Path to your first CSV file (required)
CSV_FILE_1 = r"C:\Users\achai\Desktop\skripsie- deep reinforcement learning\Data\reward_dqn.csv"

# Path to a second CSV file (optional).
# Leave as None if you only want to plot one file.
CSV_FILE_2 = r"C:\Users\achai\Desktop\skripsie- deep reinforcement learning\Data\reward_dqn_opt.csv"

# Column to use for the x-axis (must exist in the CSV file(s))
X_COL = "Step"

# Column(s) to use for the y-axis.
# Use a list even for a single column, e.g. ["reward"]
Y_COLS = ["Value"]

# Optional labels for each file's data in the legend
LABEL_1 = "DQN PreHypertuning"
LABEL_2 = "DQN PostHyperTuning"

# Type of plot: "line", "scatter", or "bar"
KIND = "line"

# Title for the plot
TITLE = "Cumulative Reward for Deep Q-Network Agents"

# If you want to save the plot to a file, set a path here, e.g. "output.png"
# Leave as None to skip saving.
SAVE_PATH = None

# ============================================================
# You shouldn't need to edit anything below this line
# ============================================================


def load_csv(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Error: file not found: {path}")
    except pd.errors.EmptyDataError:
        sys.exit(f"Error: file is empty: {path}")
    except pd.errors.ParserError as e:
        sys.exit(f"Error: could not parse CSV file: {e}")

    if df.empty:
        sys.exit(f"Error: CSV file contains no data: {path}")

    return df


def check_columns(df, path, x_col, y_cols):
    columns = list(df.columns)
    missing = [c for c in [x_col] + y_cols if c not in columns]
    if missing:
        sys.exit(
            f"Error: column(s) {missing} not found in {path}. "
            f"Available columns: {columns}"
        )


def plot_series(ax, df, x_col, y_cols, kind, label_prefix):
    for y_col in y_cols:
        label = f"{label_prefix} - {y_col}" if len(y_cols) > 1 else label_prefix
        if kind == "line":
            ax.plot(df[x_col], df[y_col], marker=".", markersize=2, linewidth=2, label=label)
        elif kind == "scatter":
            ax.scatter(df[x_col], df[y_col], label=label)
        elif kind == "bar":
            ax.bar(df[x_col].astype(str), df[y_col], label=label, alpha=0.7)


def main():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Load and plot first file
    df1 = load_csv(CSV_FILE_1)
    check_columns(df1, CSV_FILE_1, X_COL, Y_COLS)
    plot_series(ax, df1, X_COL, Y_COLS, KIND, LABEL_1)

    # Load and plot second file, if provided
    if CSV_FILE_2:
        df2 = load_csv(CSV_FILE_2)
        check_columns(df2, CSV_FILE_2, X_COL, Y_COLS)
        plot_series(ax, df2, X_COL, Y_COLS, KIND, LABEL_2)

    ax.set_xlabel(X_COL)
    ax.set_ylabel(", ".join(Y_COLS))
    ax.set_title(TITLE)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if KIND == "bar":
        plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=150)
        print(f"Plot saved to {SAVE_PATH}")

    plt.show()


if __name__ == "__main__":
    main()