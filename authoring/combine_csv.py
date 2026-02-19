import pandas as pd
import os


input_dir = "results_feb18_1150"

# Create a writer object
with pd.ExcelWriter(f'{input_dir}/combined_files.xlsx') as writer:
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            # Read the CSV
            df = pd.read_csv(f'{input_dir}/{file}')
            # Use the filename (minus .csv) as the sheet name
            sheet_name = file[:-4]
            df.to_excel(writer, sheet_name=sheet_name, index=False)