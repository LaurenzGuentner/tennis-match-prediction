# %%
### Clean data and bring it into usable form

import pandas as pd
import os # We'll use this library to handle file paths robustly
import math
import numpy as np
import re
from tqdm import tqdm
from datetime import timedelta


def load_and_clean_data(start_year,end_year):
    print("Starting the data loading process...")

    # --- Step 1: Define the years and the path to your data ---
    target_years = range(start_year, end_year + 1)

    data_directory = '../tennis_data/ATP_data/'

    # --- Step 2: Loop through the years and load each file ---
    # Create an empty list to hold the DataFrame for each year
    yearly_dfs = []

    for year in target_years:
        # Construct the full path to the file for the current year
        file_path = os.path.join(data_directory, f'{year}.csv')
        
        try:
            # Read the CSV file into a temporary DataFrame
            temp_df = pd.read_csv(file_path)
            # Add the loaded DataFrame to our list
            yearly_dfs.append(temp_df)
            print(f"Successfully loaded {year}.csv")
            
        except FileNotFoundError:
            # If a file for a specific year doesn't exist, print a warning and continue
            print(f"Warning: File for {year}.csv not found at {file_path}. Skipping.")

    # --- Step 3: Combine all yearly DataFrames into one ---
    if yearly_dfs:
        # pd.concat is the function that stacks all the DataFrames in our list together
        full_df = pd.concat(yearly_dfs, ignore_index=True)

        print("\n✅ All files have been loaded and combined successfully!")
        print(f"The DataFrame has {full_df.shape[0]} rows (matches) and {full_df.shape[1]} columns.")
        
        # --- Convert the 'tourney_date' column ---
        # '%Y' corresponds to the 4-digit year.
        # '%m' corresponds to the 2-digit month.
        # '%d' corresponds to the 2-digit day.
        full_df['tourney_date'] = pd.to_datetime(full_df['tourney_date'], format='%Y%m%d')

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
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

        print("✅ Statistical columns successfully converted!")

        # As a final cleaning step, we can drop any rows that are missing crucial data
        # that would make feature calculation impossible later on.
        print("\nDropping rows with missing essential data (like rank or stats)...")
        full_df.dropna(subset=numeric_cols, inplace=True)
        full_df.dropna(subset=['surface'], inplace=True)
        

        #Also drop the carpet matches since they are irrelevant for the modern game.
        full_df = full_df[full_df['surface'] != 'Carpet'].copy()

        print(f"The cleaned DataFrame now has {full_df.shape[0]} rows.")

        # Display the first few rows to verify everything looks correct
        print("\nHere's a preview of the combined data:")
        pd.set_option('display.max_columns', None)
    
    else:
        print("\n❌ No data files were found. Please check your 'data_directory' path.")


    return full_df


# mapping for the men's level codes you provided
TOURNEY_LENGTH_BY_LEVEL_MEN = {
    'G': 14,   # Grand Slam
    'M': 9,    # Masters 1000 typical (we'll special-case draw_size==96)
    'A': None, # ambiguous: pick by draw_size fallback
    'C': 7,    # Challenger
    'S': 7,    # Satellite / ITF
    'F': 8,    # Finals / season-ending
    'D': 7     # Davis Cup (flag as special)
}

def infer_tourney_length_men(draw_size, tourney_level):
    """
    Deterministic mapping using your men-level codes.
    Returns (tourney_length_days, uncertain_flag)
    """
    uncertain = False

    if pd.isna(tourney_level) or str(tourney_level).strip() == '':
        # no level provided -> fallback to draw-size rules but mark uncertain
        uncertain = True
        if not pd.isna(draw_size):
            ds = int(draw_size)
            if ds >= 128:
                return 14, uncertain
            if ds >= 96:
                return 13, uncertain
            if ds >= 64:
                return 9, uncertain
            return 7, uncertain
        return 7, uncertain

    lvl = str(tourney_level).strip().upper()

    # special case: masters with 96 draw (Indian Wells, Miami)
    if (not pd.isna(draw_size)) and int(draw_size) == 96:
        return 13, False

    if lvl in TOURNEY_LENGTH_BY_LEVEL_MEN:
        val = TOURNEY_LENGTH_BY_LEVEL_MEN[lvl]
        if val is None:
            # 'A' ambiguous: choose by draw size if available, else default 7 but mark uncertain
            if not pd.isna(draw_size):
                ds = int(draw_size)
                if ds >= 64:
                    return 9, False
                return 7, False
            # ambiguous, default and mark uncertain
            return 7, True
        # normal case
        # For Davis Cup ('D') we return 7 but caller can treat it specially via a flag
        return val, False

    # fallback: draw-size-based heuristic
    uncertain = True
    if not pd.isna(draw_size):
        ds = int(draw_size)
        if ds >= 128:
            return 14, uncertain
        if ds >= 96:
            return 13, uncertain
        if ds >= 64:
            return 9, uncertain
        return 7, uncertain
    return 7, uncertain


def round_to_offset_biased(round_label, draw_size, tourney_length):
    """
    Determine an offset (0..tourney_length-1) for a round label.
    Bias later rounds toward the end (non-linear).
    """
    if pd.isna(round_label) or str(round_label).strip() == '':
        return 0
    rl = str(round_label).upper().strip()

    # named rounds
    if rl in ('F', 'FINAL'):
        return tourney_length - 1
    if rl in ('SF', 'SEMI', 'SEMI-FINAL', 'SEMI_FINAL'):
        return max(tourney_length - 3, 0)
    if rl in ('QF', 'QUARTER', 'QUARTER-FINAL'):
        return max(tourney_length - 6, 0)


    # Try R<number> pattern like R128, R64, R32, R16
    early_frac=0.2
    m = re.search(r'R(\d+)', rl)
    if m:
        try:
            match_size = int(m.group(1))
            # compute number of rounds R = log2(draw_size) if possible
            if not pd.isna(draw_size):
                try:
                    R = int(round(math.log2(int(draw_size))))
                except Exception:
                    R = None
            else:
                R = None

            if R is None or R <= 1:
                # fallback small offset near beginning (draw/round ambiguous)
                return int(round((tourney_length - 1) * early_frac))

            # compute round_index and frac in [0,1]
            round_index = int(R - math.log2(match_size) + 1)
            frac = float((round_index - 1) / max(R - 1, 1))
            # clip to [0,1] to avoid negative or >1 values
            frac = max(0.0, min(1.0, frac))
            # bias transform that keeps frac in [0,1]
            frac_bias = frac ** 1.15
            frac_bias = max(0.0, min(1.0, frac_bias))
            offset = int(round(frac_bias * (tourney_length - 1)))
            return offset
        except Exception:
            # anything unexpected -> safe fallback
            return int(round((tourney_length - 1) * early_frac))

    # pattern '1R','2R' etc.
    m2 = re.search(r'(\d+)R', rl)
    if m2:
        rn = int(m2.group(1))
        if not pd.isna(draw_size):
            try:
                match_size = int(draw_size) // (2 ** (rn - 1))
                return round_to_offset_biased(f"R{match_size}", draw_size, tourney_length)
            except Exception:
                pass

    # unknown round -> small offset
    return int((tourney_length - 1) * 0.2)


def compute_approx_match_date_men(df,
                                  monday_col='tourney_date',
                                  draw_col='draw_size',
                                  round_col='round',
                                  level_col='tourney_level'):
    """
    Input: df must contain tourney monday, draw_size (optional), round (string), tourney_level (your codes).
    Output: df with columns:
      - approx_match_date (datetime)
      - tourney_length_days (int)
      - approx_offset_days (int)
      - approx_date_uncertain (bool)
    """
    df = df.copy()
    df[monday_col] = pd.to_datetime(df[monday_col])
    approx_dates, lengths, offsets, unc_flags = [], [], [], []

    for _, row in df.iterrows():
        monday = row[monday_col]
        draw = row.get(draw_col, np.nan)
        rlabel = row.get(round_col, None)
        level = row.get(level_col, None)

        tlen, uncertain = infer_tourney_length_men(draw, level)
        off = round_to_offset_biased(rlabel, draw, tlen)
        approx = monday + pd.Timedelta(days=int(off))

        approx_dates.append(approx)
        lengths.append(int(tlen))
        offsets.append(int(off))
        unc_flags.append(bool(uncertain or (level == 'D')))  # mark Davis Cup as uncertain/special

    out = df.copy()
    out['tourney_length_days'] = lengths
    out['approx_offset_days'] = offsets
    out['approx_match_date'] = pd.to_datetime(approx_dates)
    out['approx_date_uncertain'] = unc_flags
    return out

### Write ELO functions to generate player ELO's

# ELO update function using logistic distribution
def update_elo(elo_winner, elo_loser, k_factor=20):
    """
    Updates ELO ratings for a winner and loser.
    """
    expected_win = 1 / (1 + 10**((elo_loser - elo_winner) / 400))
    
    # Calculate the change in ELO
    change_in_elo = k_factor * (1 - expected_win)
    
    # Update ratings
    new_elo_winner = elo_winner + change_in_elo
    new_elo_loser = elo_loser - change_in_elo
    
    return new_elo_winner, new_elo_loser

# Calculate ELO for every player and every match

def calculate_ELO_for_df(df):

    # Initialize a dictionary to store the current ELO of each player
    elo_ratings = {}
    elo_ratings_Clay = {}
    elo_ratings_Grass = {}
    elo_ratings_Hard = {}
    STARTING_ELO = 1500

    # Lists to store the calculated pre-match ELO ratings
    winner_elos = []
    winner_elos_Clay = []
    winner_elos_Grass = []
    winner_elos_Hard = []
    loser_elos = []
    loser_elos_Clay = []
    loser_elos_Grass = []
    loser_elos_Hard = []

    # Add (surface -  relevent ELO dictionary) dictionary
    elo_dictionaries = {
        'Clay': elo_ratings_Clay, 'Grass': elo_ratings_Grass, 'Hard': elo_ratings_Hard
    } 
    
    print("Calculating ELO ratings for all matches...")
    # Loop through every match in chronological order
    for index, match in tqdm(df.iterrows()):
        winner_name = match['winner_name']
        loser_name = match['loser_name']
        match_surface = match['surface']
        
        # Look-up Step 
        # Get the current ELO for both players. If a player is new, assign the starting ELO.
        winner_pre_match_elo = elo_ratings.get(winner_name, STARTING_ELO)
        loser_pre_match_elo = elo_ratings.get(loser_name, STARTING_ELO)

        winner_pre_match_elo_Clay = elo_ratings_Clay.get(winner_name, STARTING_ELO)
        loser_pre_match_elo_Clay = elo_ratings_Clay.get(loser_name, STARTING_ELO)

        winner_pre_match_elo_Grass = elo_ratings_Grass.get(winner_name, STARTING_ELO)
        loser_pre_match_elo_Grass = elo_ratings_Grass.get(loser_name, STARTING_ELO)

        winner_pre_match_elo_Hard = elo_ratings_Hard.get(winner_name, STARTING_ELO)
        loser_pre_match_elo_Hard = elo_ratings_Hard.get(loser_name, STARTING_ELO)

        
        # Store these pre-match ratings as our features for this row
        winner_elos.append(winner_pre_match_elo)
        loser_elos.append(loser_pre_match_elo)

        winner_elos_Clay.append(winner_pre_match_elo_Clay)
        loser_elos_Clay.append(loser_pre_match_elo_Clay)

        winner_elos_Grass.append(winner_pre_match_elo_Grass)
        loser_elos_Grass.append(loser_pre_match_elo_Grass)

        winner_elos_Hard.append(winner_pre_match_elo_Hard)
        loser_elos_Hard.append(loser_pre_match_elo_Hard)

        # Add (surface -  relevent prematch ELO) dictionary
        pre_match_elos = {
        'Clay': [winner_pre_match_elo_Clay,loser_pre_match_elo_Clay], 'Grass': [winner_pre_match_elo_Grass,loser_pre_match_elo_Grass], 
        'Hard': [winner_pre_match_elo_Hard,loser_pre_match_elo_Hard]
        }

        
        # Update Step
        # Calculate the new ELO ratings after the match
        new_winner_elo, new_loser_elo = update_elo(winner_pre_match_elo, loser_pre_match_elo)
        new_winner_elo_surface, new_loser_elo_surface = update_elo(pre_match_elos[match_surface][0], pre_match_elos[match_surface][1])
        
        # Save the new ratings back to our dictionary for the next match
        elo_ratings[winner_name] = new_winner_elo
        elo_ratings[loser_name] = new_loser_elo
        elo_dictionaries[match_surface][winner_name] = new_winner_elo_surface
        elo_dictionaries[match_surface][loser_name] = new_loser_elo_surface


    return winner_elos, loser_elos, winner_elos_Clay, loser_elos_Clay, winner_elos_Grass, loser_elos_Grass, winner_elos_Hard, loser_elos_Hard

def get_recent_matches(player_name, cutoff_date, df, time_window_days):
    # ---
    # Outputs recent matches for a specific player within a specified timeframe ending at a specified cutoff date
    # ---

    player_matches = df[
        (df['winner_name'] == player_name) |
        (df['loser_name'] == player_name)
    ]

    start_date = cutoff_date - timedelta(days=time_window_days)
    recent_matches = player_matches[
        (player_matches['approx_match_date'] >= start_date) &
        (player_matches['approx_match_date'] < cutoff_date)
    ].copy()


    return recent_matches

def get_stats_from_df(filtered_df,player_name): 
        # ---
        # Takes as input a dataframe that must be already filtered for the specific players matches
        # Outputs player specific stats which are specified in the dictionary
        # ---
        
        # Handle edge case of empty data frame
        if filtered_df.empty:
            return pd.Series({
                'win_pc': 0.0,
                'matches_played': 0,
                'ace_ratio': 0.0,
                'df_ratio': 0.0,
                'ace_vs_df_ratio': 0.0,
                '1st_serve_in_pc': 0.0,
                '1st_serve_win_pc': 0.0,
                '2nd_serve_win_pc': 0.0,
                'return_win_pc': 0.0,
                'bp_save_pc': 0.0,
                'bp_conversion_pc': 0.0,
                'tiebreak_win_pc': 0.0,
                'win_pc_vs_top10': 0.0
            })

        df = filtered_df.copy()     #Work on copy to avoid warnings related to pandas slices

        # Get stats for player and create new columns to store
        is_win = (df['winner_name'] == player_name).values
        df['aces'] = np.where(is_win, df['w_ace'], df['l_ace'])
        df['dfs'] = np.where(is_win, df['w_df'], df['l_df'])
        df['svpt'] = np.where(is_win, df['w_svpt'], df['l_svpt'])
        df['first_in'] = np.where(is_win, df['w_1stIn'], df['l_1stIn'])
        df['first_won'] = np.where(is_win, df['w_1stWon'], df['l_1stWon'])
        df['second_won'] = np.where(is_win, df['w_2ndWon'], df['l_2ndWon'])
        df['bp_saved'] = np.where(is_win, df['w_bpSaved'], df['l_bpSaved'])
        df['bp_faced'] = np.where(is_win, df['w_bpFaced'], df['l_bpFaced'])
        df['opp_svpt'] = np.where(is_win, df['l_svpt'], df['w_svpt'])
        df['opp_svpts_won'] = np.where(is_win, 
                                    df['l_1stWon'] + df['l_2ndWon'], 
                                    df['w_1stWon'] + df['w_2ndWon'])
        df['return_pts_won'] = df['opp_svpt'] - df['opp_svpts_won']
        df['break_opportunities'] = np.where(is_win, df['l_bpFaced'], df['w_bpFaced'])
        df['bp_won'] = np.where(is_win, df['l_bpFaced'] - df['l_bpSaved'], df['w_bpFaced'] - df['w_bpSaved'])

        # Calculate the aggregated metrics
        # TIE-BREAK WIN PC
        # Filter for matches that included a tiebreak
        tiebreak_matches = df[df['score'].str.contains('7-6|6-7', na=False)]
        
        if not tiebreak_matches.empty:
            # Check who won in those specific tiebreak matches
            tb_wins = (tiebreak_matches['winner_name'] == player_name).sum()
            tiebreak_win_pc = tb_wins / len(tiebreak_matches)
        else:
            tiebreak_win_pc = 0.0 # No tiebreaks played

        #WIN PC VS TOP10
        # Create new column with opponent rank to filter for top ten matches
        df.loc[:, 'opponent_rank'] = np.where(is_win, df['loser_rank'], df['winner_rank'])
        top_10_matches = df[df['opponent_rank'] <= 10]
        
        if not top_10_matches.empty:
            top_10_wins = np.nansum((top_10_matches['winner_name'] == player_name))
            win_pc_vs_top10 = top_10_wins / top_10_matches.shape[0]
        else:
            win_pc_vs_top10 = 0.0 # No matches against top 10 in this period

        # Per match ratios
        df['ace_ratio'] = np.where(df['svpt'] > 0, df['aces'] / df['svpt'], 0)
        df['df_ratio'] = np.where(df['svpt'] > 0, df['dfs'] / df['svpt'], 0)
        df['ace_vs_df_ratio'] = df['aces']/(df['dfs'] + 1)
        df['1st_serve_in_pc'] = np.where(df['svpt'] > 0, df['first_in']/df['svpt'], 0)
        df['1st_serve_win_pc'] = np.where(df['first_in'] > 0, df['first_won'] / df['first_in'], 0)
        df['2nd_serve_win_pc'] = np.where((df['svpt'] - df['first_in']) > 0, df['second_won'] / (df['svpt'] - df['first_in']), 0)
        df['return_win_pc'] = np.where(df['opp_svpt'] > 0, df['return_pts_won'] / df['opp_svpt'], 0)
        df['bp_save_pc'] = np.where(df['bp_faced'] > 0, df['bp_saved'] / df['bp_faced'], 1)
        df['bp_conversion_pc'] = np.where(df['break_opportunities'] > 0, df['bp_won'] / df['break_opportunities'], 0 )


        stats = {
            'win_pc': is_win.mean(),
            'matches_played': len(df),
            'ace_ratio': df['ace_ratio'].mean(),
            'df_ratio': df['df_ratio'].mean(),
            'ace_vs_df_ratio': df['ace_vs_df_ratio'].mean(),
            '1st_serve_in_pc': df['1st_serve_in_pc'].mean(), 
            '1st_serve_win_pc': df['1st_serve_win_pc'].mean(),
            '2nd_serve_win_pc': df['2nd_serve_win_pc'].mean(),
            'return_win_pc': df['return_win_pc'].mean(),
            'bp_save_pc': df['bp_save_pc'].mean(),
            'bp_conversion_pc': df['bp_conversion_pc'].mean(),
            'tiebreak_win_pc': tiebreak_win_pc,
            'win_pc_vs_top10': win_pc_vs_top10
        }

        return pd.Series(stats)


def get_overall_form(player_name, cutoff_date, df, time_window_days):
     # --- 
     # Calculates player stats for given data frame over specified time window on all surfaces
     # ---
     
     recent_matches = get_recent_matches(player_name,cutoff_date,df,time_window_days)

     return get_stats_from_df(recent_matches,player_name)

def get_surface_form(player_name, cutoff_date, match_surface, df, time_window_days):
     # --- 
     # Calculates player stats for given data frame over specified time window on a specific surface
     # ---
    
    recent_matches = get_recent_matches(player_name, cutoff_date, df, time_window_days)
    surface_matches = recent_matches[recent_matches['surface'] == match_surface].copy()

    return get_stats_from_df(surface_matches, player_name)


def _get_fatigue_stats_for_player(player_name, cutoff_date, df):
    """
    Private helper function to calculate all fatigue and rust stats for one player.
    """
    # 1. Filter for the longest time window ONCE
    matches_last_30d = get_recent_matches(player_name, cutoff_date, df, time_window_days = 30)
    
    # 2. Find smaller windows from the already-filtered data
    window_14d = cutoff_date - timedelta(days=14)
    matches_last_14d = matches_last_30d[matches_last_30d['approx_match_date'] >= window_14d]
    
    window_7d = cutoff_date - timedelta(days=7)
    matches_last_7d = matches_last_14d[matches_last_14d['approx_match_date'] >= window_7d]
    
    # 3. Calculate "days since last match" (rust feature)
    if matches_last_30d.empty:
        days_since_last_match = 90 # Assign a default large value
    else:
        last_match_date = matches_last_30d['approx_match_date'].max()
        days_since_last_match = (cutoff_date - last_match_date).days
        
    # 4. Return all stats in a dictionary
    return {
        'matches_last_7d': len(matches_last_7d),
        'minutes_on_court_last_7d': matches_last_7d['minutes'].sum(),
        'matches_last_14d': len(matches_last_14d),
        'minutes_on_court_last_14d': matches_last_14d['minutes'].sum(),
        'matches_last_30d': len(matches_last_30d),
        'minutes_on_court_last_30d': matches_last_30d['minutes'].sum(),
        'days_since_last_match': days_since_last_match
    }

def create_static_features(df_to_process):
    # ---
    # Outputs a data frame with all static and head to head features
    # ---

    feature_rows = []

    for index, match in tqdm(df_to_process.iterrows()):
        # Alphabetical assignment for P1/P2 and getting static features
        if match['winner_name'] < match['loser_name']:
            p1_name, p2_name = match['winner_name'], match['loser_name']
            p1_rank, p2_rank = match['winner_rank'], match['loser_rank']
            p1_age, p2_age = match['winner_age'], match['loser_age']
            p1_height, p2_height = match['winner_ht'], match['loser_ht']
            target = 1
        else:
            p1_name, p2_name = match['loser_name'], match['winner_name']
            p1_rank, p2_rank = match['loser_rank'], match['winner_rank']
            p1_age, p2_age = match['loser_age'], match['winner_age']
            p1_height, p2_height = match['loser_ht'], match['winner_ht']
            target = 0


        feature_rows.append({
            'p1_rank': p1_rank, 'p2_rank': p2_rank, 'rank_diff': p1_rank - p2_rank,
            'p1_age': p1_age, 'p2_age': p2_age, 'age_diff': p1_age - p2_age,
            'p1_height': p1_height, 'p2_height': p2_height, 'height_diff': p1_height - p2_height, 
            'surface': match['surface'], 'round': match['round'],
            'target': target
        })
    
    feature_rows = pd.DataFrame(feature_rows)

    # 1. Define all possible categories for 'surface' and 'round'.
    #    Make sure these lists contain every category your model was trained on.
    all_surfaces = ['Clay', 'Grass', 'Hard']
    all_rounds = ['BR', 'F', 'QF','R128','R16', 'R32', 'R64', 'RR', 'SF'] # Example for Grand Slams

    # 2. Convert the columns to a categorical type using the full list of categories.
    #    This tells pandas about all possible categories that should exist.
    feature_rows['surface'] = pd.Categorical(feature_rows['surface'], categories=all_surfaces)
    feature_rows['round'] = pd.Categorical(feature_rows['round'], categories=all_rounds)

    # Convert surface and round info to binary data
    feature_rows = pd.get_dummies(feature_rows, columns=['surface', 'round'], prefix=['surface', 'round'])
    
    return feature_rows

def create_h2h_features(df_to_process, historical_df):
    
    feature_rows = []
    surfaces = ['Clay', 'Grass', 'Hard']

    for index, match in tqdm(df_to_process.iterrows()):
        # Alphabetical assignment for P1/P2 and getting static features
        if match['winner_name'] < match['loser_name']:
            p1_name, p2_name = match['winner_name'], match['loser_name']
        else: 
             p1_name, p2_name = match['loser_name'], match['winner_name']
        
        # Get H2H features
        h2h_matches = historical_df[
            ((historical_df['winner_name'] == p1_name) & (historical_df['loser_name'] == p2_name)) |
            ((historical_df['winner_name'] == p2_name) & (historical_df['loser_name'] == p1_name))
        ]
        h2h_matches_before = h2h_matches[h2h_matches['approx_match_date'] < match['approx_match_date']]
        
        #Calculate basic stats
        def h2h_stats(h2h_matches_before):
            p1_h2h_wins = h2h_matches_before[h2h_matches_before['winner_name'] == p1_name].shape[0]
            h2h_matches_played = h2h_matches_before.shape[0]
            p2_h2h_wins = h2h_matches_played - p1_h2h_wins
            p1_h2h_win_pc = p1_h2h_wins/h2h_matches_played if h2h_matches_played != 0 else 0.5
            p2_h2h_win_pc = p2_h2h_wins/h2h_matches_played if h2h_matches_played != 0 else 0.5
            diff_h2h_win_pc = p1_h2h_win_pc - p2_h2h_win_pc

            return [p1_h2h_wins, p2_h2h_wins, h2h_matches_played, p1_h2h_win_pc, p2_h2h_win_pc, diff_h2h_win_pc]

        general_stats = h2h_stats(h2h_matches_before)

        # Add stats to dictionary
        features = {
            'p1_h2h_wins': general_stats[0], 'p2_h2h_wins': general_stats[1], 'h2h_matches_played': general_stats[2],
            'p1_h2h_win_pc': general_stats[3], 'p2_h2h_win_pc': general_stats[4], 'diff_h2h_win_pc': general_stats[5]
        }

        # Loop through all surfaces to create same stats
        features_surface = {}
        for surface in surfaces:
            h2h_matches_surface = h2h_matches_before[h2h_matches_before['surface'] == surface]
            

            for idx,stat in enumerate(features.keys()):
                features_surface[f'{stat}_{surface}'] = h2h_stats(h2h_matches_surface)[idx]
            
        features.update(features_surface)

        feature_rows.append(features)

    return pd.DataFrame(feature_rows)
        

def create_general_dynamic_features(df_to_process, historical_df):
    # ---
    # Outputs a data frame with all dynamic player features over all surfaces and different time windows
    # ---

    feature_rows = []
    time_windows = [90, 180, 360]

    for index, match in tqdm(df_to_process.iterrows()):
        # Alphabetical assignment for P1/P2
        if match['winner_name'] < match['loser_name']:
            p1_name, p2_name = match['winner_name'], match['loser_name']
        else:
            p1_name, p2_name = match['loser_name'], match['winner_name']

        # Getting rolling features for time frame and add it to dictionary
        row_features = {}
        for window in time_windows:
            p1_form = get_overall_form(p1_name, match['approx_match_date'], historical_df, window)
            p2_form = get_overall_form(p2_name, match['approx_match_date'], historical_df, window)

            # Calculate differences in player stats
            for stat, val in p1_form.items(): row_features[f'p1_{stat}_{window}d'] = val
            for stat, val in p2_form.items(): row_features[f'p2_{stat}_{window}d'] = val
            for stat in p1_form.index:
                row_features[f'diff_{stat}_{window}d'] = p1_form[stat] - p2_form[stat]

        feature_rows.append(row_features)
    return pd.DataFrame(feature_rows)


def create_surface_dynamic_features(df_to_process, historical_df):
    # ---
    # Outputs a data frame with all dynamic player features for the specific surface of the match and different time windows
    # ---

    feature_rows = []
    time_windows = [90, 180, 360]
    surfaces_to_calculate = ['Hard', 'Clay', 'Grass']

    for index, match in tqdm(df_to_process.iterrows()):
        # Alphabetical assignment for P1/P2
        if match['winner_name'] < match['loser_name']:
            p1_name, p2_name = match['winner_name'], match['loser_name']
        else:
            p1_name, p2_name = match['loser_name'], match['winner_name']

        row_features = {}

        for window in time_windows:
            # Inner loop to calculate stats for each surface
            for surface in surfaces_to_calculate:
                p1_form = get_surface_form(p1_name, match['approx_match_date'], surface, historical_df, window)
                p2_form = get_surface_form(p2_name, match['approx_match_date'], surface, historical_df, window)

                # Add a descriptive suffix, e.g., '_Hard'
                p1_form = p1_form.add_suffix(f'_{surface}')
                p2_form = p2_form.add_suffix(f'_{surface}')

                # Add the stats to our main feature dictionary for the row
                for stat, val in p1_form.items():
                    row_features[f'p1_{stat}_{window}d'] = val
                for stat, val in p2_form.items():
                    row_features[f'p2_{stat}_{window}d'] = val
                for stat in p1_form.index:
                    if stat in p2_form.index:
                        row_features[f'diff_{stat}_{window}d'] = p1_form[stat] - p2_form[stat]

        feature_rows.append(row_features)
    return pd.DataFrame(feature_rows)


def create_fatigue_features(df_to_process, historical_df):
    # --- 
    # Outputs a dataframe with fatigue features, i.e. number of matches in past 14 days and number of matches in last 30 days
    # ---

    feature_rows = []

    for index, match in tqdm(df_to_process.iterrows()):
        # Alphabetical assignment for P1/P2
        if match['winner_name'] < match['loser_name']:
            p1_name, p2_name = match['winner_name'], match['loser_name']
        else:
            p1_name, p2_name = match['loser_name'], match['winner_name']

        # Call the helper function once for each player
        p1_stats = _get_fatigue_stats_for_player(p1_name, match['approx_match_date'], historical_df)
        p2_stats = _get_fatigue_stats_for_player(p2_name, match['approx_match_date'], historical_df)
        
        # Assemble the final feature row
        feature_row = {}
        for stat_name, p1_val in p1_stats.items():
            p2_val = p2_stats[stat_name]
            feature_row[f'p1_{stat_name}'] = p1_val
            feature_row[f'p2_{stat_name}'] = p2_val
            feature_row[f'diff_{stat_name}'] = p1_val - p2_val
        
        feature_rows.append(feature_row)

    return pd.DataFrame(feature_rows)
    

def create_ELO_features(df_to_process):
    feature_rows = []

    for index, match in tqdm(df_to_process.iterrows()):
           # Alphabetical assignment for P1/P2
            if match['winner_name'] < match['loser_name']:
                p1_name, p2_name = match['winner_name'], match['loser_name']
                p1_ELO, p2_ELO = match['winner_ELO'], match['loser_ELO']
                p1_ELO_clay, p2_ELO_clay = match['winner_ELO_clay'], match['loser_ELO_clay']
                p1_ELO_grass, p2_ELO_grass = match['winner_ELO_grass'], match['loser_ELO_grass']
                p1_ELO_hard, p2_ELO_hard = match['winner_ELO_hard'], match['loser_ELO_hard']
            else:
                p1_name, p2_name = match['loser_name'], match['winner_name']
                p1_ELO, p2_ELO = match['loser_ELO'], match['winner_ELO']
                p1_ELO_clay, p2_ELO_clay = match['loser_ELO_clay'], match['winner_ELO_clay']
                p1_ELO_grass, p2_ELO_grass = match['loser_ELO_grass'], match['winner_ELO_grass']
                p1_ELO_hard, p2_ELO_hard = match['loser_ELO_hard'], match['winner_ELO_hard']


            feature_rows.append({
                'p1_ELO': p1_ELO, 'p2_ELO': p2_ELO, 'diff_ELO': p1_ELO - p2_ELO,
                'p1_ELO_clay': p1_ELO_clay, 'p2_ELO_clay': p2_ELO_clay, 'diff_ELO_clay': p1_ELO_clay - p2_ELO_clay,
                'p1_ELO_grass': p1_ELO_grass, 'p2_ELO_grass': p2_ELO_grass, 'diff_ELO_grass': p1_ELO_grass - p2_ELO_grass,
                'p1_ELO_hard': p1_ELO_hard, 'p2_ELO_hard': p2_ELO_hard, 'diff_ELO_hard': p1_ELO_hard - p2_ELO_hard
            })

    return pd.DataFrame(feature_rows)
