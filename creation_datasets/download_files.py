from pandas import read_csv
from urllib.request import urlretrieve
from os import listdir, mkdir, makedirs, path as os_path
from tqdm import tqdm
from time import sleep
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("csv_name",
                    help="Write the name of the csv (with the extension) from which you wish to download the files")
args = parser.parse_args()

csv_name = args.csv_name

# download the 50 (can be modified) first files for each specie of the csv
# during the train phase we used 50 files, and during the test phase we used 5 files

df = read_csv("creation_datasets/fichiers_csv/"+csv_name)

print(len(df["en"].unique()), "different species")

makedirs('creation_datasets/audio_files_'+csv_name[:-4], exist_ok='True')
audio_dir = 'creation_datasets/audio_files_'+csv_name[:-4]+'/'

nots = []

nombre_audio = {}
for specie in df["en"].unique():
    nombre_audio[specie] = 0

for row in tqdm(df.iterrows(), total=df.shape[0]):

    url = row[1]["file"]
    specie = row[1]["en"]

    # change here to modify the number of files to download for each specie
    if nombre_audio[specie] < 50:

        f_name = audio_dir + str(row[1]["id"]) + row[1]["extension"]
        print("Downloading", f_name)
        nombre_audio[specie] += 1

        try:
            response = requests.get(url)
            open(f_name, 'wb').write(response.content)

        except Exception as e:
            print("\nRetrying:", url)
            print(e)
            sleep(1)
            try:
                response = requests.get(url)
                open(f_name, 'wb').write(response.content)
            except Exception as ee:
                print("Not downloaded|", f_name)
                nots.append(row[1]["id"])
                pass

if len(nots) > 0:
    with open('not_downloaded.txt', 'w') as f:
        for item in nots:
            f.write(str(item) + '\n')
    print(str(nots))
else:
    print('All files were successfully downloaded!')
