import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

current_folder = Path.cwd()
data_file = current_folder / "plot_data_pp.csv" # Both N and M are positive
#data_file = current_folder / "plot_data_pm.csv" # N is positive and M is negative

data = pd.read_csv(data_file)

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(data['t'], data['N_target'], color='blue')
ax1.scatter(data['t'][::100], data['N_pred'][::100], color='blue', s=30)
ax1.set_xlabel('Time (t)', fontsize=14)
ax1.set_ylabel('Normal force [N]', color='blue', fontsize=14)
ax1.tick_params(axis="both", labelsize=14)

ax2 = ax1.twinx()
ax2.plot(data['t'], data['M_target'], color='red')
ax2.scatter(data['t'][::100], data['M_pred'][::100], color='red', s=30)
ax2.set_ylabel('Moment [Nm]', color='red', fontsize=14)
ax2.tick_params(axis="y", labelsize=14)

plt.grid()
plt.savefig(current_folder /'plot.pdf')