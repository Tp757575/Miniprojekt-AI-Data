import pandas as pd

# Indlæs datasættet (erstat 'dataset.csv' med den faktiske fil)
df = pd.read_csv("dataset.csv")

# Vis en oversigt over de første rækker
print(df.head())

# Identificer manglende værdier
missing_values = df.isnull().sum()
print("Manglende værdier per kolonne:\n", missing_values)

# Visualiser manglende værdier
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Visualisering af manglende værdier i datasættet")
plt.show()
