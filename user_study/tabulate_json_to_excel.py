import os
import json
import pandas as pd
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
# Directory containing the JSON files
DATA_DIR = Path('data')
OUTPUT_FILE = 'study2_results_tabulated.xlsx'

# Define a map to rename the shortened shape names.
# Please complete/update the values in this dictionary with the actual text.
SHAPE_MAP = {
    "x": "Letter X",
    "y": "Blue Emoji",
    "s": "Letter S",
    "a": "Arrow",
    "e": "Emoji",
}

# Define a map to rename the numerical conditions to text.
# Please complete/update the values in this dictionary with the actual text.
CONDITION_MAP = {
    "1": "0 mm",
    "2": "3 mm",
    "3": "10.1 mm",
    "4": "30 mm",
    "5": "100 mm",
}

# =============================================================================
# Data Loading and Processing
# =============================================================================
def load_data():
    all_rows = []
    demo_rows = []
    q1_rows = []
    
    # Iterate through all JSON files in the data directory
    for file_path in DATA_DIR.glob('participant_*_study2_results.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        participant_id = data.get('participantId')
        
        # Demographics
        demo_rows.append({
            'Participant': participant_id,
            'Age': data.get('age'),
            'Gender': data.get('gender')
        })
        
        # Q1 of postQuestions
        post_questions = data.get('postQuestions', {})
        q1_rows.append({
            'Participant': participant_id,
            'Q1': post_questions.get('q1')
        })
        
        for trial in data.get('results', []):
            raw_shape = trial.get('shape')
            repetition = trial.get('repetition')
            duration = trial.get('duration')
            ratings = trial.get('ratings', {})
            
            # Use mapped names, fallback to raw name if not found in map
            mapped_shape = SHAPE_MAP.get(raw_shape, f"Unknown_{raw_shape}")
            
            row = {
                'Participant': participant_id,
                'Shape': mapped_shape,
                'Repetition': repetition,
            }
            
            # Map condition ratings
            for raw_cond, rating in ratings.items():
                mapped_cond = CONDITION_MAP.get(raw_cond, f"Cond_{raw_cond}")
                row[mapped_cond] = rating
                
            if duration is not None:
                row['Duration'] = duration
                
            all_rows.append(row)
            
    return pd.DataFrame(all_rows), pd.DataFrame(demo_rows), pd.DataFrame(q1_rows)

def main():
    if not DATA_DIR.exists():
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    df, df_demo, df_q1 = load_data()
    
    if df.empty:
        print("No data found in the JSON files.")
        return

    # Sort data for better readability (optional)
    df = df.sort_values(by=['Participant', 'Shape', 'Repetition'])
    if not df_demo.empty:
        df_demo = df_demo.sort_values(by=['Participant'])
    if not df_q1.empty:
        df_q1 = df_q1.sort_values(by=['Participant'])

    # Reorder columns to ensure Participant, Shape, Repetition, Duration are first,
    # followed by the mapped conditions in order.
    condition_cols = list(CONDITION_MAP.values())
    # Only keep condition cols that actually exist in the dataframe to avoid errors
    existing_cond_cols = [c for c in condition_cols if c in df.columns]
    
    cols = ['Participant', 'Shape', 'Repetition']
    if 'Duration' in df.columns:
        cols.append('Duration')
    cols.extend(existing_cond_cols)
        
    df = df[cols]

    # Create Excel writer
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        # 1. The first sheet has all the conditions listed (all data combined)
        df.to_excel(writer, sheet_name='All Shapes and Conditions', index=False)
        
        # 2. A sheet for each shape that reports the rating for each condition
        # Get unique shapes present in the dataframe
        unique_shapes = df['Shape'].unique()
        
        for shape in unique_shapes:
            # Filter data for this specific shape
            shape_df = df[df['Shape'] == shape].copy()
            
            # We can drop the 'Shape' column since the sheet name already indicates the shape
            shape_df = shape_df.drop(columns=['Shape'])
            
            # Ensure sheet name is valid (max 31 chars for Excel) and clean
            sheet_name = str(shape)[:31]
            # Replace invalid excel sheet name chars if any
            for char in ['\\', '/', '*', '?', ':', '[', ']']:
                sheet_name = sheet_name.replace(char, '')
                
            shape_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
        # 3. Demographics
        if not df_demo.empty:
            df_demo.to_excel(writer, sheet_name='Demographics', index=False)
            
        # 4. Q1 of postQuestions
        if not df_q1.empty:
            df_q1.to_excel(writer, sheet_name='Q1 of postQuestions', index=False)
            
    print(f"Successfully generated '{OUTPUT_FILE}' with {len(unique_shapes) + 3} sheets.")

if __name__ == '__main__':
    main()
