import pandas as pd
import re  # Import the 're' module for regular expressions

# Load your CSV file into a DataFrame
file_path = 'C:\\Users\\naran\\Desktop\\Final_clean.csv'
df = pd.read_csv(file_path, low_memory=False)


# Function to parse 'Ram and Storage' into 'Storage' and 'RAM'
def parse_memory_details(df, combined_column):
    if combined_column in df.columns:
        # Extract the numeric values for Storage and RAM using a regex pattern
        parsed_data = df[combined_column].str.extract(r'(?P<Storage>\d+)GB\s+(?P<RAM>\d+)GB RAM', flags=re.IGNORECASE)

        # Convert the extracted strings to numeric values
        df['Storage'] = pd.to_numeric(parsed_data['Storage'], errors='coerce')
        df['RAM'] = pd.to_numeric(parsed_data['RAM'], errors='coerce')


# Specify the column with the combined data
combined_column = 'Ram and Storage'

# Apply the function to split Storage and RAM
parse_memory_details(df, combined_column)

# Drop rows where 'Storage' or 'RAM' is NaN
df_cleaned = df.dropna(subset=['Storage', 'RAM'])

# Display the first few rows of the cleaned DataFrame for verification
print(df_cleaned[['Ram and Storage', 'Storage', 'RAM']].head())

# Save the cleaned DataFrame, removing rows with missing data, to a new CSV file
output_file_path = 'C:\\Users\\naran\\Desktop\\Final_clean_cleaned.csv'
df_cleaned.to_csv(output_file_path, index=False)

print(f"The cleaned CSV file has been saved to {output_file_path}")
