import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Loading Data

players = pd.read_csv("players.csv")
valuations = pd.read_csv("player_valuations.csv")
appearances = pd.read_csv("appearances.csv")

#Fixing dates
players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')
valuations['date'] = pd.to_datetime(valuations['date'], errors='coerce')
appearances['date'] = pd.to_datetime(appearances['date'], errors='coerce')


#Latest Player Valuations Transfermarkt
valuations_latest = (valuations.sort_values('date').drop_duplicates('player_id', keep='last'))

#Merging Players and valuations 
df = players.merge(valuations_latest[['player_id', 'market_value_in_eur', 'dtae']], on='player_id', how='inner')

#Player Age Engineering
df['age'] = (df['date'] - df['date_of_borth']).dt.days // 365

#Aggregating(combining) appearances
agg = appearances.groupby('player_id').agg({'goals': 'sum', 'assists': 'sum', 'minutes_played': 'sum', 'yellow_cards': 'sum', 'appearance_id': 'count'}).reset.index()

agg.rename(columns={'appearance_id': 'total_appearance'}, inplace=True)

#Per Game Stats (fixing division by 0)
agg['goals_per_game'] = agg['goals'] / agg['total_appearances'].replace(0, np.nan)
agg['assists_per_game'] = agg['assists'] / agg['total_appearances'].replace(0, np.nan)
agg['minutes'] = agg['minutes'] / agg['total_appearances'].replace(0, np.nan)
agg['yellows_per_game'] = agg['yellow_cards'] / agg['total_appearances'].replace(0, np.nan)

agg.fillna(0, inplace=True)

#merging data
df = df.merge(agg, on='player_id', how='inner')

#Select Features
df = df[['position', 'sub_position', 'foot', 'height_in_cm', 'age', 'current_club_domestic_competition_id', 'goals_per_game', 'assists_per_game', 'minutes_per_game', 'yellows_per_game', 'total_appearances', 'market_value_in_eur']]

#Data Cleanup
for col in ['position', 'sub_position', 'foot']:
    df[col] = df[col].astype(str).str.lower().str.strip()

#One Hot Encoding
df_encoded = pd.get_dummies(df, columns=['position', 'sub_position', 'foot'], drop_first= True)

#Final Dataset
df_encoded.to_csv("player_attributes.csv", index=False)

print("File Saved")