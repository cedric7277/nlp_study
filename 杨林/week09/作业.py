import matplotlib.pyplot as plt


def main():
    # =========================
    # 1. 实验数据
    # =========================
    methods = [
        "transformers\nserial",
        "transformers\nbatch=8",
        "vLLM\ncontinuous\nbatching",
    ]

    # 50 个请求总耗时，单位：秒
    total_time = [62.8, 13.1, 1.1]

    # 每秒请求数 QPS
    requests_per_sec = [0.8, 3.8, 43.6]

    # 生成吞吐，单位：tokens/sec
    tokens_per_sec = [58, 283, 3043]

    # 配色
    colors = ["#A9B7C5", "#7EA6EA", "#66E0A3"]

    # =========================
    # 2. 创建画布
    # =========================
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    fig.suptitle(
        "vLLM vs Transformers: Throughput Benchmark (Qwen2-0.5B, RTX 4060 8GB)",
        fontsize=16,
        fontweight="bold",
        y=0.98
    )

    # =========================
    # 3. 左图：总耗时
    # =========================
    ax = axes[0]

    bars = ax.bar(
        methods,
        total_time,
        color=colors
    )

    ax.set_title(
        "Total Time for 50 Requests",
        fontsize=14
    )

    ax.set_ylabel(
        "Time (seconds)",
        fontsize=12
    )

    ax.set_ylim(0, 66)

    ax.bar_label(
        bars,
        labels=[f"{v:.1f}s" for v in total_time],
        padding=3,
        fontsize=11
    )

    # =========================
    # 4. 中图：QPS
    # =========================
    ax = axes[1]

    bars = ax.bar(
        methods,
        requests_per_sec,
        color=colors
    )

    ax.set_title(
        "Requests Per Second (higher is better)",
        fontsize=14
    )

    ax.set_ylabel(
        "QPS (requests/sec)",
        fontsize=12
    )

    ax.set_ylim(0, 46)

    ax.bar_label(
        bars,
        labels=[f"{v:.1f}" for v in requests_per_sec],
        padding=3,
        fontsize=11
    )

    # =========================
    # 5. 右图：Token 吞吐量
    # =========================
    ax = axes[2]

    bars = ax.bar(
        methods,
        tokens_per_sec,
        color=colors
    )

    ax.set_title(
        "Generation Throughput (tokens/sec)",
        fontsize=14
    )

    ax.set_ylabel(
        "Tokens / sec (generated)",
        fontsize=12
    )

    ax.set_ylim(0, 3200)

    ax.bar_label(
        bars,
        labels=[f"{v}" for v in tokens_per_sec],
        padding=3,
        fontsize=11
    )

    # =========================
    # 6. 统一样式
    # =========================
    for ax in axes:
        ax.tick_params(
            axis="x",
            labelsize=11
        )

        ax.tick_params(
            axis="y",
            labelsize=10
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    # 调整布局
    plt.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

    # =========================
    # 7. 保存图片
    # =========================
    output_file = "vllm_vs_transformers_benchmark.png"

    plt.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    print(
        f"图片已生成：{output_file}"
    )

    # 显示图片
    plt.show()


if __name__ == "__main__":
    main()
