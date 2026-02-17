import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_misc = pd.read_csv("MiscNatPopTari.csv")
df_coduri = pd.read_csv("CoduriTariExtins.csv")


# A1. Salvarea in fisierul Cerinta1.csv a tarilor cu spor natural negativ
df_cerinta1 = df_misc[df_misc["RS"] < 0]
df_cerinta1.to_csv("Cerinta1.csv", index=False)

# A2. Salvarea in fisierul Cerinta2.csv a valorilor medii la nivel de continent pentru indicatorii de mai sus
df_merged = pd.merge(df_misc, df_coduri, on="Country Number")
indicatori = ["RS", "FR", "IM", "MMR", "LE", "LEM", "LEF"]
df_cerinta2 = df_merged.groupby("Continent")[indicatori].mean()
df_cerinta2.to_csv("Cerinta2.csv")


# B1. Variantele componentelor principale
X = df_misc[indicatori].values
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
R = np.corrcoef(X_std, rowvar=False)
val_proprii, vect_proprii = np.linalg.eig(R)

index_sortare = np.argsort(val_proprii)[::-1]
val_proprii = val_proprii[index_sortare]
vect_proprii = vect_proprii[:, index_sortare]

print("Variantele componentelor principale (Eigenvalues):")
print(val_proprii)

# B2. Scorurile asociate instantelor. Scorurile vor fi salvate in fisierul Scoruri.csv
scoruri = X_std @ vect_proprii

nume_instante = df_misc["Country Name"].values
nume_componente = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

df_scoruri = pd.DataFrame(data=scoruri)
df_scoruri.index = nume_instante
df_scoruri.columns = nume_componente
df_scoruri.to_csv("Scoruri.csv")

# B3. Graficul scorurilor in primele doua axe principale

plt.figure(figsize=(10, 10))

plt.scatter(scoruri[:,0],scoruri[:,1], c='blue', alpha=0.7)

plt.title("Graficul scorurilor în primele două axe principale (C1 și C2)")
plt.xlabel("C1")
plt.ylabel("C2")
plt.grid(True)
plt.show()