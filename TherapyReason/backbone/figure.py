import matplotlib.pyplot as plt

import numpy as np

results = {
    "Best Baseline": [0.781, 0.906, 0.635, 0.485, 0.069, 1.90, 0.743, 0.805, 0.569, 0.419, 0.083, 1.84, 0.627, 0.617, 0.371, 0.240, 0.039, 2.42,
                      0.580, 0.552, 0.345, 0.222, 0.033, 2.80],
    "default: StageNet+DeepSeek": [0.593, 0.802, 0.657, 0.506, 0.091, 1.64, 0.596, 0.679, 0.586, 0.434, 0.112, 1.65, 0.410, 0.445, 0.397, 0.260,
                                   0.055,
                                   2.05, 0.432, 0.385, 0.370, 0.243, 0.045, 2.04],
    "ML: GRU": [0.622, 0.758, 0.656, 0.506, 0.088, 1.57, 0.564, 0.697, 0.577, 0.423, 0.117, 1.75, 0.379, 0.432, 0.376, 0.242, 0.058, 2.17, 0.380,
                0.383, 0.348, 0.224, 0.049, 2.19],
    "ML: TRANS": [0.638, 0.741, 0.658, 0.510, 0.087, 1.51, 0.570, 0.703, 0.583, 0.430, 0.116, 1.73, 0.399, 0.441, 0.389, 0.253, 0.057, 2.12, 0.489,
                  0.340, 0.370, 0.246, 0.039, 1.69],
    # "LLM: Qwen3-8B": [0.570, 0.694, 0.600, 0.442, 0.092, 1.64, 0.581, 0.610, 0.562, 0.407, 0.098, 1.63, 0.362, 0.421, 0.356, 0.227, 0.063, 2.25,
    #                   0.381, 0.348, 0.326, 0.209, 0.050, 2.17],
    "LLM: Qwen3-235B": [0.609, 0.733, 0.640, 0.485, 0.083, 1.58, 0.589, 0.627, 0.575, 0.420, 0.097, 1.62, 0.305, 0.632, 0.391, 0.252, 0.083, 2.05,
                        0.301, 0.569, 0.362, 0.234, 0.073, 2.09],
    "LLM: Kimi": [0.543, 0.895, 0.642, 0.497, 0.112, 1.78, 0.589, 0.627, 0.573, 0.418, 0.097, 1.62, 0.373, 0.448, 0.384, 0.249, 0.059, 2.26, 0.361,
                  0.382, 0.341, 0.219, 0.051, 2.33],
}
fontsize = 14

# 处理数据：计算每个数据集第3、4位置metric的平均值
processed_data = {}
for method, values in results.items():
    avg_metrics = []
    # 每个数据集有6个metric，共4个数据集
    for i in range(4):
        start_idx = i * 6
        metric3 = values[start_idx + 2]
        metric4 = values[start_idx + 3]
        avg = (metric3 + metric4) / 2
        avg_metrics.append(avg)
    processed_data[method] = avg_metrics

methods = [m for m in processed_data.keys() if m != "Best Baseline"]
baseline_values = processed_data["Best Baseline"]
colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

if __name__ == '__main__':
    fig, axes = plt.subplots(2, 2, figsize=(8, 4))

    dataset_names = ["MIMIC-III Drug Rec.", "MIMIC-IV Drug Rec.", "MIMIC-III Diagnosis Pred.", "MIMIC-IV Diagnosis Pred."]

    for i, ax in enumerate(axes.flat):
        current_values = [processed_data[method][i] for method in methods]

        x = np.arange(len(methods))
        bars = ax.bar(x, current_values, color=colors, alpha=0.7)

        ax.axhline(y=baseline_values[i], color='r', linestyle='--', linewidth=2, label='Best Baseline')

        ax.set_title([r'$\bf(a)$ ', r'$\bf(b)$ ', r'$\bf(c)$ ', r'$\bf(d)$ '][i] + dataset_names[i],
                     fontsize=fontsize, y=-0.23)
        if i in [0, 2]:
            ax.set_ylabel("Avg. F1/Jaccard" + r'$\uparrow$', fontsize=fontsize - 2)
        ax.set_xticks(x)
        ax.set_xticklabels([])

        if i == 3:
            y_min = min(min(current_values), baseline_values[i]) * 0.9
        else:
            y_min = min(min(current_values), baseline_values[i]) * 0.95
        y_max = max(max(current_values), baseline_values[i]) * 1.01
        ax.set_ylim(y_min, y_max)

    from matplotlib.patches import Rectangle

    fill = Rectangle((0, 0), 1, 1, fill=False, edgecolor='none')
    handles = []
    for method, color in zip(methods, colors):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, label=method))
    handles.append(plt.Line2D([0], [0], color='r', linestyle='--', label='Best Baseline'))

    # 0: default, 1: gru, 2: trans, 3: qwen8b, 4: qwen235b, 5: kimi, 6: baseline
    # 0: default, 1: gru, 2: trans, 3: qwen235b, 4: kimi, 5: baseline
    handles = [handles[i] if i is not None else fill for i in [0, 3, 1, 4, 2, 5]]
    fig.legend(handles=handles, loc='upper center', ncol=3,
               bbox_to_anchor=(0.5, 1.0), fontsize=fontsize - 2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.07, hspace=0.2)
    plt.savefig("Backbone.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
