import pandas as pd
from odf.opendocument import load
from pandas_ods_reader import read_ods
import re

def is_mostly_javascript(text):
    """
    Basic check to see if the text is mostly JavaScript.
    This function can be enhanced based on specific requirements.
    """
    javascript_patterns = [
        r'function\s+\w+\s*\(',  # Function declarations
        r'var\s+\w+\s*=',       # Variable declarations
        r'if\s*\(.*\)\s*{',     # If statements
        r'\w+\s*:\s*\w+',       # Object properties
        # Add more patterns as needed
    ]
    text = str(text)
    matches = [bool(re.search(pattern, text)) for pattern in javascript_patterns]
    return any(matches)

def remove_invalid_rows_ods(file_path, text_column, output_file_path):
    # Load the spreadsheet
    sheet_idx = 1  # Assuming you want to process the first sheet
    df = read_ods(file_path, sheet_idx)

    # Count total rows before removal
    total_rows = len(df)

    # Remove rows where all elements are NaN
    df_cleaned = df.dropna(how='all')
    empty_rows_removed = total_rows - len(df_cleaned)

    # Count JavaScript rows and remove them
    javascript_rows = df_cleaned[text_column].apply(is_mostly_javascript).sum()
    df_cleaned = df_cleaned[~df_cleaned[text_column].apply(is_mostly_javascript)]

    # Save the cleaned DataFrame back to .ods
    df_cleaned.to_excel(output_file_path, engine='odf', index=False)

    return empty_rows_removed, javascript_rows

def save_rows_with_matching_column_value(input_file_path, output_file_path, column_name, matching_value):
    """
    Save rows with a matching column value to another file.

    :param input_file_path: Path to the input file (can be .csv, .xls, .xlsx, .ods, etc.)
    :param output_file_path: Path to save the output file (recommend the same format as input)
    :param column_name: Name of the column to match the value
    :param matching_value: The value to match in the specified column
    """
    # Load the data
    df = pd.read_excel(input_file_path)

    specific_urls = ["https://www.globallogic.com/services","https://www.globallogic.com/about"]

    # Filter rows where the column value matches
    filtered_df = df[df[column_name].isin(matching_value)]
    
    # Further filter rows that contain any of the specified URLs
    filtered_df = filtered_df[filtered_df['url'].apply(lambda x: any(url in x for url in specific_urls))]

    # Select only 'text' and 'url' columns
    filtered_df = filtered_df[['metadata/title', 'text', 'url']]

    # Save the filtered rows to a new file
    filtered_df.to_csv(output_file_path, index=False)  # Use to_excel or to_csv as needed


def top_rows_with_max_data(file_path, text_column, top_n=50):
    # Load the spreadsheet
    sheet_idx = 1  # Assuming you want to process the first sheet
    df = read_ods(file_path, sheet_idx)

    # Calculate the length of text in the specified column (excluding spaces)
    df['text_length'] = df[text_column].apply(lambda x: len(str(x).replace(' ', '')) if pd.notnull(x) else 0)

    # Sort the DataFrame based on the text length and select the top N rows
    top_rows = df.sort_values(by='text_length', ascending=False).head(top_n)

    return top_rows.drop(columns=['text_length'])



# Replace with your file paths
input_file_path = 'dataset_website-content-crawler.xlsx'
text_column_name = 'text'
output_file_path = 'dataset_website-content-crawler-en.xlsx'



#Data Curation
'''
empty_rows_removed, javascript_rows_removed = remove_invalid_rows_ods(input_file_path, text_column_name, output_file_path)
print(f"Empty rows removed: {empty_rows_removed}")
print(f"Rows with mostly JavaScript content removed: {javascript_rows_removed}")
'''

#Pick only rows in english
save_rows_with_matching_column_value(input_file_path, 'dataset_website-content-crawler-en.ods', 'metadata/languageCode',['en','en-us'] )

#Top 50 rows
# Get the top 50 rows
#top_rows = top_rows_with_max_data(output_file_path, 'text_column_name', 10)

# Save the top 50 rows to a new CSV file
#top_rows.to_csv('top-rows.csv', index=False)
