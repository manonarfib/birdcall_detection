import pandas as pd
import requests
from tqdm import tqdm
import argparse

# writes in the csv the informations to download the files for all species of a country with more than 150 (can be modified) recordings

parser = argparse.ArgumentParser()
parser.add_argument(
    "country", help="The country for which you want to create the database. You may chose among this list : ['Arab Emirates', 'Algeria', 'Andorra', 'Angola', 'Antarctica', 'Argentina', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Belize', 'Bhutan', 'Bolivia', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 'Burkina Faso', 'Cambodia', 'Canada', 'Cape Verde', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Congo (Brazzaville)', 'Congo (Democratic Republic)', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 'Denmark', 'Dominican Republic', 'East Timor', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Estonia', 'Ethiopia', 'Finland', 'France', 'French Guiana', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Ireland', 'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kyrgyzstan', 'Laos', 'Latvia', 'Liberia', 'Lithuania', 'Macedonia', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Malta', 'Mexico', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Nigeria', 'Norway', 'Oman', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'Sao Tome', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'South Africa', 'South Korea', 'Spain', 'Sri Lanka', 'Suriname', 'Sweden', 'Switzerland', 'Taiwan', 'Tajikistan', 'Tanzania', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela', 'Vietnam', 'Zambia', 'Zimbabwe']")
args = parser.parse_args()

country = args.country
country = country.replace(" ", "_")

df = pd.DataFrame()

response = requests.get(
    f'https://xeno-canto.org/api/2/recordings?query=cnt:{country}')
js = response.json()
ids, files, extensions, ens, lengths, gens, cnts = [], [], [], [], [], [], []
for n_page in tqdm(range(1, js["numPages"]+1)):
    response = requests.get(
        f'https://xeno-canto.org/api/2/recordings?query=cnt:{country}&page={n_page}')
    page_js = response.json()
    for recording in page_js["recordings"]:
        if not pd.isnull(recording["file-name"]):
            ids.append(recording["id"])
            files.append(recording["file"])
            extensions.append(recording["file-name"][-4:].lower())
            ens.append(recording["en"])
            lengths.append(recording["length"])
            gens.append(recording["gen"])
            cnts.append(recording["cnt"])
df_ = pd.DataFrame.from_records(
    {'id': ids, "file": files, "extension": extensions, "en": ens, "gen": gens, "length": lengths, "cnt": cnts})
df = df.append(df_)

# removing recordings that are not classified
df = df[df['en'] != 'Identity unknown'].copy()
df = df[df['en'] != 'Soundscape'].copy()

counts = df['en'].value_counts()


# change the value if you wish to keep more or less species
chosen = counts[counts >= 150].index
df = df[df["en"].isin(chosen)]

df.to_csv("creation_datasets/fichiers_csv/"+country+".csv", index=False)
