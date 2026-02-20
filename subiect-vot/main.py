import matplotlib.pyplot as plt
import pandas as pd

df_vot=pd.read_csv("Vot.csv")
df_coduri=pd.read_csv("Coduri.csv")



#A1 Salvarea in fisierul Cerinta1.csv a categoriei de alegatori pentru care s a inregistrat cel mai mic procent de prezenta la vot. Se va salva codul Siruta,numele localitatii si categoria de alegatori.
df_categorii=df_vot[categorii]
df_vot['Categorie']=df_categorii.idxmin(axis=1)
df_cerinta1=df_vot[['Siruta','Localitate','Categorie']]
df_cerinta1.to_csv("Cerinta1.csv",index=False)



#A2
df_merged=pd.merge(df_cerinta1,df_coduri,on='Siruta')
df_cerinta2=df_merged.groupby(['County'])[categorii].mean()
df_cerinta2.to_csv("Cerinta2.csv",index=False)



#B1 Sa se efectueze analiza factoriala, fara rotatie de factori, pentru structura votului si sa se furnizeze urmatoarele rezultate:
# Se va calcula si se va afisa pragul de semnificatie asociat respingerii/acceptarii testului p-value.

X=df_vot[categorii].values
chi_patrat,p_value=calculate_bartlett_sphericity(df_vot[categorii])
print(f"Pragul de semnificatie Bartlett {p_value}")

#B2 Scorurile factoriale. Vor fi salvate in fisierul f.csv

fa=FactorAnalyzer(n_factor=2,rotation=None)
fa.fit(X)
scoruri=fa.transform(df_vot[df_categorii])

df_f=pd.DataFrame(scoruri)
df_f.index=df_vot['Localitate'].values
df_f.columns=['F1','F2']
df_f.to_csv("F1.csv",index=False)
#B3 Graficul scorurilor factoriale pentru primii doi factori.


plt.figure(figsize=(10,10))
plt.scatter(scoruri[:,0],scoruri[:,1],color='red',alpha=0.5)

plt.title("Graficul scorurilor factoriale")
plt.xlabel("F1")
plt.ylabel("F2")
plt.grid(True)
plt.show()