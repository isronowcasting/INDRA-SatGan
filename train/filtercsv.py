import pandas as pd
import os

def filter_sequences_by_year_and_month(input_csv_path, output_csv_path, target_years, target_months):
    """
    Reads a CSV of file sequences and keeps only those sequences where ALL files
    belong to the same target year AND are within the target months.
    Assumes filename format YYYYMMDDHHMM.HDF5.

    Args:
        input_csv_path (str): Path to the original CSV file.
        output_csv_path (str): Path where the new, filtered CSV will be saved.
        target_years (list): A list of valid years as strings (e.g., ['2022', '2023']).
        target_months (list): A list of valid month numbers as strings (e.g., ['06', '07', '08', '09']).
    """
    print(f"Reading sequences from: {input_csv_path}")
    
    try:
        df = pd.read_csv(input_csv_path, header=None)
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {input_csv_path}")
        return
    except pd.errors.EmptyDataError:
        print("Input CSV is empty. No output will be generated.")
        return

    initial_row_count = len(df)
    print(f"Found {initial_row_count} total sequences to process.")

    valid_rows = []

    for index, row in df.iterrows():
        sequence_files = row.dropna().tolist()
        
        if not sequence_files:
            continue

        is_sequence_valid = True
        first_year = None

        for i, filename in enumerate(sequence_files):
            filename_str = str(filename)
            
            # Basic format check
            if len(filename_str) < 8:
                print(f"Warning: Filename '{filename_str}' in row {index} is malformed. Discarding sequence.")
                is_sequence_valid = False
                break
            
            try:
                # Extract year and month from YYYYMMDD... format
                year_str = filename_str[0:4]
                month_str = filename_str[4:6]
                
                # 1. Check if the year is in our target list
                if year_str not in target_years:
                    is_sequence_valid = False
                    break

                # 2. Check if the month is in our target list
                if month_str not in target_months:
                    is_sequence_valid = False
                    break

                # 3. Check if all files in the sequence are from the SAME year
                if i == 0:
                    first_year = year_str  # Set the year for this sequence
                elif year_str != first_year:
                    is_sequence_valid = False
                    break

            except (IndexError, TypeError):
                print(f"Warning: Could not parse filename '{filename_str}' in row {index}. Discarding sequence.")
                is_sequence_valid = False
                break
        
        # If after checking all files, the sequence is still valid, keep it
        if is_sequence_valid:
            valid_rows.append(row)

    if not valid_rows:
        print("No valid sequences found for the specified years and months. Output file will be empty.")
        filtered_df = pd.DataFrame()
    else:
        filtered_df = pd.DataFrame(valid_rows)

    final_row_count = len(filtered_df)
    print(f"\nKept {final_row_count} sequences.")
    print(f"  - All files are from years: {target_years}")
    print(f"  - All files are from months: {target_months}")
    print(f"  - All files within a sequence belong to the same year.")
    print(f"Discarded {initial_row_count - final_row_count} sequences.")

    filtered_df.to_csv(output_csv_path, header=False, index=False)
    print(f"Filtered CSV saved successfully to: {output_csv_path}")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    INPUT_CSV = "/home/sac/data_67/Nowcasting/data/input_sequences_valid_south.csv"
    
    # The new output file containing only the desired data
    OUTPUT_CSV = "/home/sac/data_67/Nowcasting/data/input_sequences_south_JJAS_strict.csv"
    
    # --- Define your strict criteria ---
    # 1. Years to keep
    TARGET_YEARS = ['2022', '2023']
    # 2. Months to keep (June, July, August, September)
    TARGET_MONTHS = ['06', '07', '08', '09']

    # Run the filtering process
    filter_sequences_by_year_and_month(INPUT_CSV, OUTPUT_CSV, TARGET_YEARS, TARGET_MONTHS)