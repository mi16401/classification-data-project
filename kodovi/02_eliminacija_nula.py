import pandas as pd

def main():
    #UCITAVANJE RANIJE SACUVANIH PODATAKA
    data = pd.read_csv('bigdata.csv')


    #PROVERA DA U MATRICI STVARNO POSTOJI KOLONA CIJE SU SVE VREDNOSTI NULA
    #proveravamo sumu vrednosti u koloni
    suma = data['hg38_A1CF'].sum()
    print(suma) #dobija se vrednost 0


    #Posto smo se uverili da takva kolona postoji, eliminisemo nula-kolone
    data = data.loc[:, (data != 0).any(axis=0)]
    print(data.shape) #Nova dimenzija matrice: (16100, 10752)

    #Cuvamo redukovanu matricu u novu datoteku kako bismo je koristili u daljem radu
    data.to_csv('data.csv')

if __name__ == "__main__":
    main()
