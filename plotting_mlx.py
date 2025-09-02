import os
import time
import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window):
    if not values:
        return []
    w = max(1, min(window, len(values)))
    arr = np.array(values, dtype=float)
    csum = np.cumsum(np.insert(arr, 0, 0.0))
    avg = (csum[w:] - csum[:-w]) / w
    return np.concatenate([[avg[0]] * (w - 1), avg])


def save_progress(
    out_dir,
    agent,
    train_returns,
    train_lengths,
    action_means,
    action_stds,
    cumulative_steps,
    start_time,
    final=False,
):
    total_steps = cumulative_steps[-1]
    elapsed = max(time.time() - start_time, 1e-9)
    threshold = -350.0
    baseline = -1600.0

    returns_ma = moving_average(train_returns, 10)

    steps_to_threshold = None
    for i, r in enumerate(returns_ma):
        if r >= threshold:
            steps_to_threshold = cumulative_steps[i + 1]
            break

    x_steps = np.array(cumulative_steps[1:], dtype=float)
    y_returns = np.array(train_returns, dtype=float)
    denom = max(1e-9, (threshold - baseline))
    norm_scores = (
        np.clip((y_returns - baseline) / denom, 0.0, 1.0) if len(y_returns) else np.array([])
    )
    auc_norm = float(np.trapz(norm_scores, x_steps)) if len(x_steps) > 1 else 0.0
    auc_norm_frac = auc_norm / x_steps[-1] if len(x_steps) > 0 else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax = axes[0, 0]
    ax.plot(range(len(train_returns)), train_returns, label="Return", color="#1f77b4")
    ax.plot(range(len(returns_ma)), returns_ma, label="Return (10MA)", color="#ff7f0e")
    ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
    ax.axhline(baseline, color="gray", linestyle=":", label="Baseline")
    ax.set_title("Return per episode")
    ax.legend()
    if steps_to_threshold is not None:
        ax.annotate(
            f"steps≈{int(steps_to_threshold)}",
            xy=(len(train_returns) - 1, returns_ma[-1]),
            xytext=(0.5, 0.1),
            textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="gray"),
            fontsize=9,
        )

    ax = axes[0, 1]
    ax.plot(range(len(train_lengths)), train_lengths, label="Episode length", color="#2ca02c")
    ax.set_title("Episode length")
    ax.legend()

    ax = axes[0, 2]
    critic_ma = moving_average(agent.critic_losses, 50)
    ax.plot(range(len(critic_ma)), critic_ma, label="Critic loss (MA)", color="#d62728")
    ax.set_title("Critic TD error")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(range(len(agent.alphas)), agent.alphas, label="alpha", color="#9467bd")
    ax2 = ax.twinx()
    alpha_loss_ma = moving_average(agent.alpha_losses, 50)
    ax2.plot(range(len(alpha_loss_ma)), alpha_loss_ma, label="alpha loss", color="#8c564b")
    ax.set_title("Entropy α")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    ax = axes[1, 1]
    gap_ma = moving_average(agent.q_gaps, 50)
    ax.plot(range(len(gap_ma)), gap_ma, label="|Q1-Q2| (MA)", color="#e377c2")
    ax.set_title("Q disagreement")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(range(len(action_means)), action_means, label="action mean", color="#17becf")
    ax.plot(range(len(action_stds)), action_stds, label="action std", color="#bcbd22")
    ax.set_title("Action stats")
    ax.legend()

    fig.suptitle(
        f"steps={total_steps}, throughput={total_steps/elapsed:.1f}/s, normAUC={auc_norm_frac:.3f}"
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.93])

    ep = len(train_returns)
    filename = (
        os.path.join(out_dir, f"progress_ep{ep:04d}.pdf") if not final else os.path.join(out_dir, "progress_mlx.pdf")
    )
    plt.savefig(filename, bbox_inches="tight")

    if final:
        plt.figure()
        plt.matshow(agent.memory.buf.T, cmap="viridis", aspect="auto")
        if agent.state_dim == 3:
            plt.yticks(
                range(8),
                [
                    "cos(theta)",
                    "sin(theta)",
                    "theta_dot",
                    "action",
                    "reward",
                    "cos(theta')",
                    "sin(theta')",
                    "theta_dot'",
                ],
            )
        else:
            plt.yticks([])
            plt.ylabel(f"Features (2×{agent.state_dim} + action + reward)")
        plt.xlabel("Experience index (time)")
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.savefig(os.path.join(out_dir, 'buffer.png'), bbox_inches="tight", dpi=300)

    plt.close('all')
