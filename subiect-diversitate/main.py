import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer


df_diversitate = pd.read_csv("Diversitate.csv")
df_coduri = pd.read_csv("Coduri_Localitati.csv")

ani = [col for col in df_diversitate.columns if col not in ['Siruta', 'Localitate']]



# A1. Să se calculeze și să se salveze în fișierul Cerinta1.csv, localitățile în care
# cel puțin pentru un an diversitatea a fost 0.
df_cerinta1 = df_diversitate[(df_diversitate[ani] == 0).any(axis=1)]
df_cerinta1.to_csv("Cerinta1.csv", index=False)

# A2. Să se determine pentru fiecare județ localitatea în care diversitatea medie este maximă.
# Rezultatul va fi salvat în fișierul Cerinta2.csv.
df_merged = pd.merge(df_coduri[['Siruta', 'Judet']], df_diversitate, on="Siruta")
df_merged["Media"] = df_merged[ani].mean(axis=1)
index_max = df_merged.groupby('Judet')["Media"].idxmax()
df_cerinta2 = df_merged.loc[index_max, ['Judet', 'Localitate', 'Media']]
df_cerinta2.columns = ["Judet", "Localitate", "Diversitate Maxima"]
df_cerinta2.to_csv("Cerinta2.csv", index=False)



X = df_diversitate[ani].values

fa = FactorAnalyzer(n_factors=3, rotation="varimax")

fa.fit(X)

# B1. Varianța factorilor comuni. Se va salva tabelul varianței în fișierul Varianta.csv.
# fa.get_factor_variance() returnează: (Varianța, Proporția varianței, Proporția cumulată)
v, pv, cv = fa.get_factor_variance()
df_varianta = pd.DataFrame({
    "Varianta": v,
    "Procent varianta": pv * 100,
    "Procent cumulat": cv * 100
})
df_varianta.to_csv("Varianta.csv", index=False)

# B2. Corelațiile factoriale (corelațiile variabile - factori comuni). Salvate în r.csv.
loadings = fa.loadings_
df_r = pd.DataFrame(data=loadings, index=ani,columns=[f"Factor_{i+1}" for i in range(loadings.shape[1])])
df_r.to_csv("r.csv")

# B3. Trasarea cercului corelațiilor pentru primii doi factori comuni

plt.figure(figsize=(10, 10))
plt.title("Cercul Corelațiilor (Factor 1 vs Factor 2)")

circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--')
plt.gca().add_patch(circle)

for i in range(len(ani)):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='blue', alpha=0.6, head_width=0.03)

plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.grid(True, linestyle=':')
plt.show()