import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('penguins_size.csv')
print(df.info())
dropRowIndicies = df[df['body_mass_g'].isnull() == True].index.tolist()
df.drop(dropRowIndicies, axis=0, inplace=True)
female_df = df[df['sex'] == 'FEMALE']
male_df = df[df['sex'] == 'MALE']

# Using the help of the body_mass_g filling the NA values in sex column. If body_mass_g => 4546 then it will be male else female.
male_filled = df[(df['sex'].isnull() == True) & (df['body_mass_g'] > 4546)].fillna('MALE')
female_filled = df[(df['sex'].isnull() == True) & (df['body_mass_g'] <= 4546)].fillna('FEMALE')
df = pd.concat([df, male_filled, female_filled], axis=0)
df = df.dropna()
df.reset_index(inplace=True)
# Checking for the '.' in the sex feature.
df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()
dotIndex = df[df['sex'] == '.'].index.tolist()
df.at[dotIndex[0], 'sex'] = 'FEMALE'
df_final = df.drop('index', axis=1)
df_final.to_csv('penguins_size_clean_data.csv', index=False)

