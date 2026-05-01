import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
players = pd.read_csv("players.csv")
valuations = pd.read_csv("player_valuations.csv")
appearances = pd.read_csv("appearances.csv")

# Fixing dates
# Dates are converted so age can be calculated and the most recent valuation can be selected.
players['date_of_birth'] = pd.to_datetime(players['date_of_birth'], errors='coerce')
valuations['date'] = pd.to_datetime(valuations['date'], errors='coerce')
appearances['date'] = pd.to_datetime(appearances['date'], errors='coerce')

# Player Age Engineering
# Age is created from date of birth since it is expected to affect market value.
players['age'] = (pd.Timestamp.today() - players['date_of_birth']).dt.days // 365

# Latest Player Valuations from Transfermarkt
# Since players can have multiple valuations, only the most recent one is kept.
valuations_latest = (
    valuations.sort_values('date')
    .drop_duplicates('player_id', keep='last')
)[['player_id', 'market_value_in_eur']]

# Remove existing market value columns from players to prevent duplicate columns after merging
players_clean = players.drop(
    columns=[col for col in players.columns if 'market_value' in col],
    errors='ignore'
)

# Merge players and valuations
# This connects each player with their most recent market value.
df = players_clean.merge(valuations_latest, on='player_id', how='inner')

# Aggregating appearances
# Appearance data has many rows per player, so it is combined into one row per player.
agg = appearances.groupby('player_id').agg(
    goals=('goals', 'sum'),
    assists=('assists', 'sum'),
    minutes_played=('minutes_played', 'sum'),
    yellow_cards=('yellow_cards', 'sum'),
    total_appearances=('appearance_id', 'count')
).reset_index()

# Per Game Stats
# Per game statistics are created so players can be compared more fairly.
agg['goals_per_game'] = agg['goals'] / agg['total_appearances'].replace(0, np.nan)
agg['assists_per_game'] = agg['assists'] / agg['total_appearances'].replace(0, np.nan)
agg['minutes_per_game'] = agg['minutes_played'] / agg['total_appearances'].replace(0, np.nan)
agg['yellows_per_game'] = agg['yellow_cards'] / agg['total_appearances'].replace(0, np.nan)

# Any missing per game values are filled with 0 because that means no available contribution in that stat.
agg.fillna(0, inplace=True)

# Merge appearance statistics into the main dataframe
df = df.merge(agg, on='player_id', how='inner')

# Select Features
# Name is kept only for readability, while the other features are used for prediction.
df = df[['name', 'position', 'sub_position', 'foot', 'height_in_cm', 'age',
         'current_club_domestic_competition_id',
         'goals_per_game', 'assists_per_game', 'minutes_per_game',
         'yellows_per_game', 'total_appearances',
         'market_value_in_eur']]

# Data Cleanup
# Text columns are cleaned so values with different capitalization or spaces are treated the same.
for col in ['position', 'sub_position', 'foot']:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Numeric fixes
# Missing numerical values are filled with the median to avoid losing too many rows.
df['age'] = df['age'].fillna(df['age'].median())
df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())

# Categorical fixes
# Missing categorical values are filled with unknown so the model can still use the row.
df['position'] = df['position'].fillna('unknown')
df['sub_position'] = df['sub_position'].fillna('unknown')
df['foot'] = df['foot'].fillna('unknown')

# Final cleanup
# Any remaining missing values are removed before saving the final dataset.
df.dropna(inplace=True)

# One Hot Encoding
# Categorical features are converted into numeric columns so they can be used by the models.
df_encoded = pd.get_dummies(
    df,
    columns=['position', 'sub_position', 'foot', 'current_club_domestic_competition_id'],
    drop_first=True
)

# Check missing values before saving
# This confirms the final dataset does not contain empty cells.
print("\nMissing Values in Final Dataset:")
print(df_encoded.isnull().sum())

print("\nTotal Missing Values:")
print(df_encoded.isnull().sum().sum())

# Final Dataset
df_encoded.to_csv("player_attributes.csv", index=False)

# Regular distribution plot
# This shows the original market value distribution, which is heavily skewed.
plt.figure()
plt.hist(df['market_value_in_eur'], bins=50)
plt.title("Distribution of Market Value")
plt.xlabel("Market Value (EUR)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("market_value_distribution.png")
plt.show()

# Log-transformed distribution plot
# This shows how the distribution would look after using log transformation.
plt.figure()
plt.hist(np.log1p(df['market_value_in_eur']), bins=50)
plt.title("Log-Transformed Distribution of Market Value")
plt.xlabel("Log Market Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("log_market_value_distribution.png")
plt.show()

print("File Saved Successfully")