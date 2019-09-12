import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np

def main():

    #racunanje kvantila i IQR-a
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    print(IQR)

    print(df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))
    df_out_iqr = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(df_out_iqr.shape) #Dimenzija matrice nakon uklanjanja outlier-a: (0, 10753)

    #uklanjamo atribut klase kako ne bi bilo problema pri poredjenju razlicitih tipova
    #(atribut 'class' je tipa string dok su svi ostali tipa int)
    data = data.iloc[:,df.columns != 'class']
    z = np.abs(stats.zscore(data))
    data_out= dfata[(z < 3).all(axis=1)]
    print(data_out.shape) #Dimenzija matrice nakon uklanjanja outlier-a: (673, 10752)

if __name__ == "__main__":
    main()
