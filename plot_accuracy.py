import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change to a relative directory from the script's location
os.chdir(script_dir)

model_names = ["slowfast_r50", "slow_r50", "res_r50"]

acc_means = []
acc_sems = []

for model_name in model_names:
    with open(f"result/confusions/{model_name}.pkl", "rb") as File:
        cm = pickle.load(File)

    acc_all_folds = []

    for fold_cm in cm:
        acc = np.diag(fold_cm) / np.sum(fold_cm, axis=1)
        acc = np.append(acc, np.mean(acc))
        acc_all_folds.append(acc)

    acc_all_folds = np.array(acc_all_folds)

    acc_means.append(np.mean(acc_all_folds, axis=0))
    acc_sems.append(stats.sem(acc_all_folds, axis=0))

acc_means = np.array(acc_means)
acc_sems = np.array(acc_sems)

# Plotting
x = np.arange(len(model_names))
width = 0.1  # Bar width

files = sorted([f for f in os.listdir("stimuli") if f.startswith("processed_")])
classnames = np.unique([f.split("_")[1] for f in files])
classnames = np.append(classnames, "overall")

fig, ax = plt.subplots()
for i in range(acc.shape[0]):
    ax.bar(
        x + i * width,
        acc_means[:, i],
        width,
        yerr=acc_sems[:, i],
        label=f"{classnames[i]}",
    )

ax.axhline(y=(6 / 36), color="black", linestyle="dashed")

ax.set_ylabel("Accuracy")
ax.set_title("Network accuracy")
ax.set_xticks(x + 3 * width)
ax.set_xticklabels(["S_wx + F", "S_nox", "S_1"])
plt.legend()
plt.tight_layout()
plt.savefig("plot/acc.png", dpi=300)
