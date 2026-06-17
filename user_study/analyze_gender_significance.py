import argparse
import textwrap

import numpy as np
import pandas as pd
from scipy import stats

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def load_data(path):
    df_ratings = pd.read_excel(path, sheet_name="All Shapes and Conditions")
    df_demo = pd.read_excel(path, sheet_name="Demographics")
    
    # Clean up column names in demo just in case
    df_demo['Gender'] = df_demo['Gender'].str.strip().str.lower()
    
    # Merge on Participant
    df = df_ratings.merge(df_demo[['Participant', 'Gender']], on='Participant', how='left')
    
    return df

def analyze_overall(df):
    error_cols = ["0 mm", "3 mm", "10.1 mm", "30 mm", "100 mm"]
    
    # Average across all conditions
    df['Average_Rating'] = df[error_cols].mean(axis=1)
    
    # Average across all repetitions/shapes to get ONE score per participant
    participant_avg = df.groupby(['Participant', 'Gender'])['Average_Rating'].mean().reset_index()
    
    males = participant_avg[participant_avg['Gender'] == 'male']['Average_Rating']
    females = participant_avg[participant_avg['Gender'] == 'female']['Average_Rating']
    
    print("\n" + "─" * 72)
    print("  OVERALL SIGNIFICANCE (All Shapes & Conditions)")
    print("─" * 72)
    print(f"  Males   (n={len(males)}): Mean = {males.mean():.2f}, SD = {males.std():.2f}")
    print(f"  Females (n={len(females)}): Mean = {females.mean():.2f}, SD = {females.std():.2f}")
    
    if len(males) == 0 or len(females) == 0:
        print("\n  Not enough data for both genders to run a statistical test.")
        return

    stat, p = stats.mannwhitneyu(males, females, alternative='two-sided')
    
    print(f"\n  Mann-Whitney U = {stat:.2f}, p-value = {p:.4f} ({sig_stars(p)})")

def analyze_per_shape(df):
    error_cols = ["0 mm", "3 mm", "10.1 mm", "30 mm", "100 mm"]
    df['Average_Rating'] = df[error_cols].mean(axis=1)
    
    shapes = sorted(df['Shape'].unique())
    
    print("\n" + "─" * 72)
    print("  SIGNIFICANCE PER SHAPE")
    print("─" * 72)
    
    for shape in shapes:
        shape_df = df[df['Shape'] == shape]
        participant_avg = shape_df.groupby(['Participant', 'Gender'])['Average_Rating'].mean().reset_index()
        
        males = participant_avg[participant_avg['Gender'] == 'male']['Average_Rating']
        females = participant_avg[participant_avg['Gender'] == 'female']['Average_Rating']
        
        print(f"\n  Shape: {shape}")
        print(f"    Males   (n={len(males)}): Mean = {males.mean():.2f}, SD = {males.std():.2f}")
        print(f"    Females (n={len(females)}): Mean = {females.mean():.2f}, SD = {females.std():.2f}")
        
        if len(males) == 0 or len(females) == 0:
            print("    Skipping test - not enough groups.")
            continue
            
        stat, p = stats.mannwhitneyu(males, females, alternative='two-sided')
        print(f"    Mann-Whitney U = {stat:.2f}, p-value = {p:.4f} ({sig_stars(p)})")

def analyze_per_condition(df):
    error_cols = ["0 mm", "3 mm", "10.1 mm", "30 mm", "100 mm"]
    
    print("\n" + "─" * 72)
    print("  SIGNIFICANCE PER MISALIGNMENT CONDITION")
    print("─" * 72)
    
    for error in error_cols:
        participant_avg = df.groupby(['Participant', 'Gender'])[error].mean().reset_index()
        
        males = participant_avg[participant_avg['Gender'] == 'male'][error]
        females = participant_avg[participant_avg['Gender'] == 'female'][error]
        
        print(f"\n  Condition: {error}")
        print(f"    Males   (n={len(males)}): Mean = {males.mean():.2f}, SD = {males.std():.2f}")
        print(f"    Females (n={len(females)}): Mean = {females.mean():.2f}, SD = {females.std():.2f}")
        
        if len(males) == 0 or len(females) == 0:
            print("    Skipping test - not enough groups.")
            continue
            
        stat, p = stats.mannwhitneyu(males, females, alternative='two-sided')
        print(f"    Mann-Whitney U = {stat:.2f}, p-value = {p:.4f} ({sig_stars(p)})")


def main(path):
    df = load_data(path)
    
    # Filter the exact same shapes as in the original analysis script
    df = df[~df['Shape'].isin(['Letter X', 'Blue Emoji'])]
    
    analyze_overall(df)
    analyze_per_shape(df)
    analyze_per_condition(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gender Significance using Mann-Whitney U test (between-subjects)"
    )
    parser.add_argument("--file", "-f", default="study2_results_tabulated.xlsx", help="Path to excel file")
    
    args = parser.parse_args()
    main(args.file)
