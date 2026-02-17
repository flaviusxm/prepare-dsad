import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Citirea fișierelor de intrare
df_indicators = pd.read_csv("GlobalIndicatorsPerCapita_2021.csv")
df_codes = pd.read_csv("CoduriTari.csv")
nume_coloane=['GNI', 'ChangesInv', 'Exports', 'Imports', 'FinalConsExp', 'GrossCF', 'HouseholdConsExp', 'AgrHuntForFish','Construction', 'Manufacturing', 'MiningManUt', 'TradeT', 'TransportComm', 'Other']
# A1
# Calculul excedentului comercial pentru fiecare tara diferenta  dintre Exports si imports. Vor fi salvaate in fisierul Cerinta 1.csv tarile
# cu excedent pozitiv, in ordinea descrescatoare a excedentului. Se va salva codul de tara numele tarii si exportul, importul si excedentul.
df_indicators["Excedent"] = df_indicators["Exports"] - df_indicators["Imports"]
df_cerinta1 = df_indicators[df_indicators["Excedent"] > 0][["CountryID", "Country", "Exports", "Imports", "Excedent"]]
df_cerinta1 = df_cerinta1.sort_values(by=["Excedent"], ascending=[False])
df_cerinta1.to_csv("Cerinta1.csv", index=False)

# A2
# Salvarea in fisierele Africa.csv,Asia.csv,... a matricelor de corelatii intre indicatori la nivelul continentelor
# Numele fisierului este numele continentului pentru care se calculeaza matricea de corelatie.
df_merged = pd.merge(df_indicators, df_codes[["CountryID", "Continent"]], on="CountryID")
for c, g in df_merged.groupby("Continent"):
    g[nume_coloane].corr().to_csv(f"{c}.csv")

# B1
# Analiza în componente principale (PCA)
X = df_indicators[nume_coloane]
X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
R = np.corrcoef(X_std, rowvar=False)

val_proprii, vect_proprii = np.linalg.eig(R)

idx = np.argsort(val_proprii)[::-1]
val_proprii = val_proprii[idx]
vect_proprii = vect_proprii[:, idx]

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(val_proprii) + 1), val_proprii, 'ro-')
plt.axhline(y=1, color='b', linestyle='--')
plt.title("Plot")
plt.xlabel("Componenta")
plt.ylabel("Valoare proprie")
plt.show()

# Corelatiile factoriale (corelatii dintre variabilele observate si componentele principale)
# Corelatiile vor fi salvate in fisierul r.csv
r = vect_proprii * np.sqrt(val_proprii)

df_r = pd.DataFrame(data=r, index=nume_coloane, columns=[f"PC{i + 1}" for i in range(len(nume_coloane))])
df_r.to_csv("r.csv")

# Valorile cosinus si comunalitatile. Aceastea vor fi salvate in fisierele cosin.csv si comm.csv
cos = r ** 2
comm = np.cumsum(cos, axis=1)

df_cosin = pd.DataFrame(data=cos, index=nume_coloane, columns=[f"PC{i + 1}" for i in range(len(nume_coloane))])
df_cosin.to_csv("cosin.csv")

df_comm = pd.DataFrame(data=comm, index=nume_coloane, columns=[f"PC{i + 1}" for i in range(len(nume_coloane))])
df_comm.to_csv("comm.csv")

# C Intr un model de analiza discriminanta liniara cu 6 variabile predictor si 4 clase, matricea de covarianta totala a fost salvata in fisierul T.csv, iar centrii de clase au fost salvati in fisierul G.csv. Predictorii sunt X1,...X6
# clasele etichetate sunt A,B,C si D. Sa se afiseze la consola clasa in care va fi clasificata instanta x=[118,19.8,2.5,8.3,61.1,8.3]

# Inversa matricei de covarianta totala va fi calculata prin functia np.linalg.inv()

df_t = pd.read_csv("T.csv", index_col=0)
df_g = pd.read_csv("G.csv", index_col=0)
x = np.array([118, 19.8, 2.5, 8.3, 61.1, 8.3])
t_inversat = np.linalg.inv(df_t.values)

scor_maxim = -9999999
clasa_predictie = ""
for c in df_g.index:
    g_k = df_g.loc[c].values
    # Formula corectă a discriminantului liniar: x' * T_inv * g_k - 0.5 * g_k' * T_inv * g_k
    scor = np.dot(np.dot(x, t_inversat), g_k) - 0.5 * np.dot(np.dot(g_k, t_inversat), g_k)
    if scor > scor_maxim:
        clasa_predictie = c
        scor_maxim = scor

print(f"Instanta x este in clasa {clasa_predictie}")