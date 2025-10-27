# %%
import requests
from typing import Dict
from datetime import timedelta
import numpy as np
import pandas as pd
from tqdm import tqdm
from data_prep_refactored import load_and_clean_data, compute_approx_match_date_men, calculate_ELO_for_df
import joblib
from data_prep_refactored import create_static_features, create_h2h_features, create_general_dynamic_features, create_surface_dynamic_features, create_fatigue_features, create_ELO_features

def fetch_json(feed_url: str, timeout: int = 10) -> Dict:
    r = requests.get(feed_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
    r.raise_for_status()
    return r.json()

start_date = '2025-08-17'
start_date = pd.to_datetime(start_date)

def create_df_from_draw_data_json(draw_data):

    # Access the relevant data within the JSON
    # Assume the match data is under a key called 'matches'

    try:
        match_data = draw_data['matches']
        
        # 3. Create a pandas DataFrame 🐼
        df = pd.DataFrame(match_data)
        
        # 4. Display the first 5 rows of the DataFrame
        print("DataFrame created successfully!")
        pd.set_option('display.max_columns', None)


    except KeyError:
        print("The key 'matches' was not found. Please inspect the available keys:")
        print(draw_data.keys())
    except TypeError:
        print("The JSON data might not be a dictionary. Let's see what it is:")
        print(type(draw_data))
        print(draw_data)

    df = df[df['status'] != 'Walkover']

    return df

def fetch_matches_for_round(draw_df, round):
    round_matches = draw_df[draw_df['roundNameShort'] == round]

    return round_matches

# possible round short hands as input: 'R1', 'R2', 'R3', 'R4', 'QF', 'SF', 'F'
def get_match_stats_for_input_df(df, round):
    round_matches = fetch_matches_for_round(df,round)

    # Reset the index of the filtered DataFrame to ensure it starts from 0.
    # drop=True prevents the old index from being added as a new column.
    round_matches = round_matches.reset_index(drop=True)

    input_df = pd.DataFrame({})

    input_df['winner_name'] = [round_matches['team1'][i]['firstNameA'] + ' ' + round_matches['team1'][i]['lastNameA'] for i in range(len(round_matches['team1']))]
    input_df['loser_name'] = [round_matches['team2'][i]['firstNameA'] + ' ' + round_matches['team2'][i]['lastNameA'] for i in range(len(round_matches['team1']))]
    input_df['approx_match_date'] = [start_date + timedelta(days=round_matches['eventDay'][i]) for i in range(len(round_matches['team1']))]
    input_df['round'] = [round for i in range(len(round_matches['roundNameShort']))]
    input_df['surface'] = ['Hard' for i in range(len(round_matches['roundNameShort']))]

    return input_df


def get_winnners_of_round(round_matches):   
    

    winner_list = []

    for id, match in tqdm(round_matches.iterrows()):
        if match['winner'] == None:
            continue
        if int(match['winner']) == 1:
            winner_list.append(round_matches['team1'][id]['firstNameA'] + ' ' + round_matches['team1'][id]['lastNameA'])
        elif int(match['winner']) == 2:
            winner_list.append(round_matches['team2'][id]['firstNameA'] + ' ' + round_matches['team2'][id]['lastNameA'])


    return winner_list

def clean_data(df):
    # --- Clean and Convert Statistical Columns to Numeric ---

    print("Converting all statistical columns to a numeric data type...")

    # Create a list of all the columns that should contain numbers
    numeric_cols = [
                'winner_rank', 'loser_rank', 'winner_age', 'loser_age',
                'w_ace', 'l_ace', 'w_df', 'l_df', 'w_svpt', 'l_svpt',
                'w_1stIn', 'l_1stIn', 'w_1stWon', 'l_1stWon', 'w_2ndWon', 'l_2ndWon',
                'w_bpSaved', 'l_bpSaved', 'w_bpFaced', 'l_bpFaced',
                'winner_ht', 'loser_ht', 'draw_size'
            ]

    # Loop through each column in our list
    for col in numeric_cols:
        # Convert the column to a numeric type.
        # The key is errors='coerce', which will replace any value that
        # cannot be converted to a number with NaN (Not a Number).
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("✅ Statistical columns successfully converted!")

    # As a final cleaning step, we can drop any rows that are missing crucial data
    # that would make feature calculation impossible later on.
    print("\nDropping rows with missing essential data (like rank or stats)...")
    df.dropna(subset=numeric_cols, inplace=True)
    df.dropna(subset=['surface'], inplace=True)

    print(f"The cleaned DataFrame now has {df.shape[0]} rows.")

    return df

### Prepare historical dataframe

def get_full_master_df(start_year, end_year, enable_update = True):

    master_df = load_and_clean_data(start_year,end_year)
    # We make sure to chronologically order the matches:
    master_df.sort_values(by=['tourney_date','tourney_id','match_num'], inplace=True)

    if enable_update == True:
        data_directory = '../tennis_data/ATP_data/'
        filename = 'us_open_update.csv' 

        us_open_update_df = pd.read_csv(data_directory + filename)
        us_open_update_df['tourney_date'] = pd.to_datetime(us_open_update_df['tourney_date'], format='%Y%m%d')

        master_df = pd.concat([master_df,us_open_update_df], axis = 0)

    ELOs = calculate_ELO_for_df(master_df)

    # Add general ELO column to data frame
    master_df['winner_ELO'] = ELOs[0]
    master_df['loser_ELO'] = ELOs[1]
    # Add ELO's for specific surfaces
    master_df['winner_ELO_clay'] = ELOs[2]
    master_df['loser_ELO_clay'] = ELOs[3]

    master_df['winner_ELO_grass'] = ELOs[4]
    master_df['loser_ELO_grass'] = ELOs[5]

    master_df['winner_ELO_hard'] = ELOs[6]
    master_df['loser_ELO_hard'] = ELOs[7]

    master_df = compute_approx_match_date_men(master_df)

    return master_df


static_stats_winner = ['winner_rank', 'winner_age', 'winner_ht', 'winner_ELO', 'winner_ELO_clay', 'winner_ELO_grass', 'winner_ELO_hard']
static_stats_loser = ['loser_rank', 'loser_age', 'loser_ht', 'loser_ELO', 'loser_ELO_clay', 'loser_ELO_grass', 'loser_ELO_hard']

# Assume 'master_df' is your full, chronologically-sorted DataFrame
# and it has been loaded and prepared.

def get_most_recent_match(player_name, df):
    """
    Finds the most recent match for a specific player from a chronological DataFrame.

    Args:
        player_name (str): The name of the player.
        df (pd.DataFrame): The DataFrame to search within (must be sorted by date).

    Returns:
        pd.Series: A Series representing the row of the most recent match, or None if not found.
    """
    # Filter for all matches involving the player
    player_matches = df[
        (df['winner_name'] == player_name) |
        (df['loser_name'] == player_name)
    ]

    # Check if the player was found
    if player_matches.empty:
        print(f"No matches found for {player_name}.")
        return None

    # The last row of the filtered DataFrame is the most recent match
    most_recent_match = player_matches.iloc[-1]
    
    return most_recent_match

    
def get_player_stats_for_input_df(input_df, master_df):
    static_features_list = []

    # Get historical stats for players
    for idx, match in tqdm(input_df.iterrows()):
        
        static_feature_row = {}

        p1_name = match['winner_name']
        p2_name = match['loser_name']

        if p1_name == 'Botic van De Zandschulp':
            p1_name = 'Botic van de Zandschulp'
        elif p2_name == 'Botic van De Zandschulp':
            p2_name = 'Botic van de Zandschulp'

        p1_last_match = get_most_recent_match(p1_name, master_df)
        p2_last_match = get_most_recent_match(p2_name, master_df)

        if p1_last_match is not None:
            if p1_name == p1_last_match['winner_name']:
                for stat in static_stats_winner:
                    static_feature_row[stat] = p1_last_match[stat]
            else: 
                for id, stat in enumerate(static_stats_loser):
                    static_feature_row[static_stats_winner[id]] = p1_last_match[stat]

        else:
        # Handle the case where the player is new and has no history
            print(f"Player {p1_name} has no match history. Using default values.")
        # You might want to fill in default stats for a new player here

        if p2_last_match is not None:
            if p2_name == p2_last_match['loser_name']:
                for stat in static_stats_loser:
                    static_feature_row[stat] = p2_last_match[stat]
            else: 
                for id, stat in enumerate(static_stats_winner):
                    static_feature_row[static_stats_loser[id]] = p2_last_match[stat]

        else:
        # Handle the case where the player is new and has no history
            print(f"Player {p1_name} has no match history. Using default values.")
        # You might want to fill in default stats for a new player here


        static_features_list.append(static_feature_row)


    static_features_df = pd.DataFrame(static_features_list)
    input_df = pd.concat([input_df,static_features_df], axis = 1)

    return input_df




# Define the filename
model_filename = 'tennis_predictor_model.joblib'
trained_model = joblib.load(model_filename)



def create_features_for_matches(matches_df, master_df):
    """
    Takes a DataFrame of upcoming matches and generates the full feature set.
    Each row in matches_df should contain 'winner_name' and 'loser_name' just because it matches the format of the training data and other info like 'round'.
    """
    # 1. Create static features (based on most recent player stats)
    #    This part requires modifying your 'get_most_recent_match' logic
    #    to handle the new format.
    static_features = create_static_features(matches_df).drop('target', axis=1)

    # 2. Create all other features
    h2h_features = create_h2h_features(matches_df, master_df)
    general_dynamic_features = create_general_dynamic_features(matches_df, master_df)
    surface_features = create_surface_dynamic_features(matches_df, master_df)
    fatigue_features = create_fatigue_features(matches_df, master_df)
    elo_features = create_ELO_features(matches_df)

    # 3. Combine all features into one DataFrame
    all_features = pd.concat([
        static_features, h2h_features, general_dynamic_features,
        surface_features, fatigue_features, elo_features
    ], axis=1)

    return all_features

# Helper function to get the latest stats for a single player
def get_player_latest_stats(player_name, master_df):
    """
    Finds the most recent stats for a player from the master DataFrame.
    Returns a dictionary of their stats with generic keys (e.g., 'rank', 'age').
    """
    # This function is from your notebook
    last_match = get_most_recent_match(player_name, master_df)
    
    if last_match is None:
        # Handle new players with no history - return default/NaN values
        print(f"Player {player_name} has no history, using default stats.")
        return {
            'rank': 150.0, 'age': 25.0, 'ht': 185.0, 'ELO': 1500,
            'ELO_clay': 1500, 'ELO_grass': 1500, 'ELO_hard': 1500
        }

    stats = {}
    if player_name == last_match['winner_name']:
        # Player was the winner in their last recorded match
        stats['rank'] = last_match['winner_rank']
        stats['age'] = last_match['winner_age']
        stats['ht'] = last_match['winner_ht']
        stats['ELO'] = last_match['winner_ELO']
        stats['ELO_clay'] = last_match['winner_ELO_clay']
        stats['ELO_grass'] = last_match['winner_ELO_grass']
        stats['ELO_hard'] = last_match['winner_ELO_hard']
    else:
        # Player was the loser in their last recorded match
        stats['rank'] = last_match['loser_rank']
        stats['age'] = last_match['loser_age']
        stats['ht'] = last_match['loser_ht']
        stats['ELO'] = last_match['loser_ELO']
        stats['ELO_clay'] = last_match['loser_ELO_clay']
        stats['ELO_grass'] = last_match['loser_ELO_grass']
        stats['ELO_hard'] = last_match['loser_ELO_hard']
        
    return stats

def visualize_tournament_tree(predictions):
    """
    Prints a text-based visualization of the predicted tournament bracket.
    """
    for round_name, round_data in predictions.items():
        # The f-string is corrected here (backslash removed)
        print(f"""
---------------------------------
|         {round_name.upper()}         |
---------------------------------""")
        
        matches = round_data['matches']
        winners = round_data['winners']
        
        for i in range(len(matches)):
            # Note: We use the original 'winner_name' and 'loser_name' columns
            # from the input DataFrame, which act as p1 and p2 placeholders.
            p1 = matches.iloc[i]['winner_name']
            p2 = matches.iloc[i]['loser_name']
            winner = winners[i]
            
            # Highlight the winner using an asterisk
            if p1 == winner:
                p1_display = f"*{p1}"
                p2_display = p2
            else:
                p1_display = p1
                p2_display = f"*{p2}"
            
            print(f"Match {i+1}: {p1_display:<25} vs. {p2_display:<25}")
            
    print("\n* = Predicted Winner")





