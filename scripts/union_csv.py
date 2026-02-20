import pandas as pd
import os

def unir_csv(carpeta_csv, nombre_de_archivo_unido = 'mallas_curriculares.csv'):
    dataframes = []
    contador = 1

    for archivos in os.listdir(carpeta_csv):
        if archivos.endswith('.csv'):
            ruta_archivo = os.path.join(carpeta_csv, archivos)
            df = pd.read_csv(ruta_archivo, on_bad_lines='warn')
            dataframes.append(df)

    if not dataframes:
        print("No hay data")
        return

    df_unido = pd.concat(dataframes, ignore_index= True)

    df_unido['id']= range(1, len(df_unido) + 1)# para que los id salgan desde el 1 al 500...

    df_unido.to_csv(nombre_de_archivo_unido, index= False)
    
    print (f"ya los archivos estan unidos en : {nombre_de_archivo_unido}")

if __name__== '__main__':
    unir_csv('csv')