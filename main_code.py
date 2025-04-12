
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.impute import SimpleImputer, KNNImputer

# Indlæser datasættet (justér stien hvis nødvendigt)
df = pd.read_csv(r"C:\Users\thoma\Desktop\python_work\Mini_projects\AI&DATA Miniprojekt\archive\horse.csv")

# Vis de første 5 rækker
print("Datasættets første 5 rækker:")
print(df.head())

# Vis beskrivende statistik for numeriske kolonner
print("\nBeskrivende statistik:")
print(df.describe())




# Tæl manglende værdier per kolonne
missing_values = df.isnull().sum()
print("Manglende værdier per kolonne:")
print(missing_values)



# Visualiser manglende data med heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Heatmap: Manglende værdier i datasættet")
plt.show()



# Visualiser korrelation mellem manglende værdier
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull().corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap: Korrelation mellem manglende værdier")
plt.show()




# Udvælg kun numeriske kolonner til imputering
num_cols = df.select_dtypes(include=np.number).columns
df_num = df[num_cols]

# Visualiser fordelingen før imputering
plt.figure(figsize=(8, 5))
sns.histplot(df_num.melt(), x="value", hue="variable", kde=True, bins=30)
plt.title("Datadistribution før imputering")
plt.show()




# Mean-imputering: Erstat manglende værdier med gennemsnit
mean_imputer = SimpleImputer(strategy="mean")
df_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(df_num), columns=num_cols)

# KNN-imputering: Brug nærmeste naboer til at estimere manglende værdier
knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_num), columns=num_cols)

# Visualiser fordeling efter hver imputeringstype
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_mean_imputed.melt(), x="value", hue="variable", kde=True, bins=30, ax=axes[0])
axes[0].set_title("Datadistribution efter Mean-imputering")
sns.histplot(df_knn_imputed.melt(), x="value", hue="variable", kde=True, bins=30, ax=axes[1])
axes[1].set_title("Datadistribution efter KNN-imputering")
plt.show()




try:
    # Opret forbindelse til SQLite-database
    conn = sqlite3.connect("medical_data.db")
    cursor = conn.cursor()

    # Opret 3 relaterede tabeller: Patients, Measurements og Diagnoses
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS Patients (
        patient_id INTEGER PRIMARY KEY,
        age INTEGER,
        gender TEXT,
        surgery TEXT
    );

    CREATE TABLE IF NOT EXISTS Measurements (
        measurement_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        heart_rate REAL,
        respiratory_rate REAL,
        temperature REAL,
        blood_pressure TEXT,
        FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
    );

    CREATE TABLE IF NOT EXISTS Diagnoses (
        diagnosis_id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        diagnosis TEXT,
        treatment TEXT,
        FOREIGN KEY (patient_id) REFERENCES Patients(patient_id)
    );
    """)
    conn.commit()
    print("Database og tabeller oprettet succesfuldt!")
except sqlite3.Error as e:
    print("Fejl i databaseoprettelse:", e)
finally:
    conn.close()
