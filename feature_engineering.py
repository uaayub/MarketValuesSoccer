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

#Player Age Engineering
players['age'] = (pd.Timestamp.today() - players['date_of_birth']).dt.days // 365

#Latest Player Valuations Transfermarkt
valuations_latest = (
    valuations.sort_values('date')
    .drop_duplicates('player_id', keep='last')
)[['player_id', 'market_value_in_eur']]

#Remove existing market value columns from players (prevents duplicates)
players_clean = players.drop(
    columns=[col for col in players.columns if 'market_value' in col],
    errors='ignore'
)

#Merging Players and valuations 
df = players_clean.merge(valuations_latest, on='player_id', how='inner')

#Aggregating(combining) appearances
agg = appearances.groupby('player_id').agg(
    goals=('goals', 'sum'),
    assists=('assists', 'sum'),
    minutes_played=('minutes_played', 'sum'),
    yellow_cards=('yellow_cards', 'sum'),
    total_appearances=('appearance_id', 'count')
).reset_index()

#Per Game Stats
agg['goals_per_game'] = agg['goals'] / agg['total_appearances'].replace(0, np.nan)
agg['assists_per_game'] = agg['assists'] / agg['total_appearances'].replace(0, np.nan)
agg['minutes_per_game'] = agg['minutes_played'] / agg['total_appearances'].replace(0, np.nan)
agg['yellows_per_game'] = agg['yellow_cards'] / agg['total_appearances'].replace(0, np.nan)

agg.fillna(0, inplace=True)

#merging data
df = df.merge(agg, on='player_id', how='inner')

#Select Features (including name for readability)
df = df[['name', 'position', 'sub_position', 'foot', 'height_in_cm', 'age',
         'current_club_domestic_competition_id',
         'goals_per_game', 'assists_per_game', 'minutes_per_game',
         'yellows_per_game', 'total_appearances',
         'market_value_in_eur']]

#Data Cleanup
for col in ['position', 'sub_position', 'foot']:
    df[col] = df[col].astype(str).str.lower().str.strip()

#numeric fixes
df['age'] = df['age'].fillna(df['age'].median())
df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())

#categorical fixes
df['position'] = df['position'].fillna('unknown')
df['sub_position'] = df['sub_position'].fillna('unknown')
df['foot'] = df['foot'].fillna('unknown')

#final cleanup
df.dropna(inplace=True)

# One Hot Encoding
df_encoded = pd.get_dummies(df, columns=['position', 'sub_position', 'foot', 'current_club_domestic_competition_id'], drop_first=True)

# CHECK MISSING VALUES BEFORE SAVING
print("\nMissing Values in Final Dataset:")
print(df_encoded.isnull().sum())

print("\nTotal Missing Values:")
print(df_encoded.isnull().sum().sum())

#Final Dataset
df_encoded.to_csv("player_attributes.csv", index=False)

# Regular distribution plot
# plt.figure()
# plt.hist(df['market_value_in_eur'], bins=50)
# plt.title("Distribution of Market Value")
# plt.xlabel("Market Value (EUR)")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("market_value_distribution.png")
# plt.show()

# Log-transformed distribution plot
# plt.figure()
# plt.hist(np.log1p(df['market_value_in_eur']), bins=50)
# plt.title("Log-Transformed Distribution of Market Value")
# plt.xlabel("Log Market Value")
# plt.ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("log_market_value_distribution.png")
# plt.show()

print("File Saved Successfully")