import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sqlite3
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

# Indlæser datasættet
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

# Udvælg kun numeriske kolonner
num_cols = df.select_dtypes(include=np.number).columns
df_num = df[num_cols]

# Imputering på den originale data
mean_imputer = SimpleImputer(strategy="mean")
df_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(df_num), columns=num_cols)

knn_imputer = KNNImputer(n_neighbors=5)
df_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(df_num), columns=num_cols)

# Skaler de tre datasæt
scaler = StandardScaler()
df_num_scaled = pd.DataFrame(scaler.fit_transform(df_num), columns=df_num.columns)
df_mean_imputed_scaled = pd.DataFrame(scaler.fit_transform(df_mean_imputed), columns=df_mean_imputed.columns)
df_knn_imputed_scaled = pd.DataFrame(scaler.fit_transform(df_knn_imputed), columns=df_knn_imputed.columns)

# Visualiser skalerede data før imputering
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_num_scaled)
plt.title("Boksplot: Skaleret datadistribution før imputering")
plt.xticks(rotation=90)
plt.show()

# Visualiser skalerede data efter Mean-imputering
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_mean_imputed_scaled)
plt.title("Boksplot: Skaleret datadistribution efter Mean-imputering")
plt.xticks(rotation=90)
plt.show()

# Visualiser skalerede data efter KNN-imputering
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_knn_imputed_scaled)
plt.title("Boksplot: Skaleret datadistribution efter KNN-imputering")
plt.xticks(rotation=90)
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
