import os
import sys
import pandas as pd
from pandas_ods_reader import read_ods
from openai import OpenAI
from config_ft import config

# Function to generate question and answer
def generate_question_and_answer(client, text, engine='text-davinci-003', q_max_tokens=50, a_max_tokens=150):
    try:
        # Generate a question
        question_prompt = f"Create a question about IT products and services based on the following information, relevant to GlobalLogic:\n{text}"
        question_response = client.completions.create(
            model=engine,
            prompt=question_prompt,
            max_tokens=q_max_tokens
        )
        question = question_response.choices[0].text.strip()

        # Generate an answer
        answer_prompt = f"Provide a detailed answer suitable for a public-facing chat agent at GlobalLogic, for the following question based on the IT services context:\nQuestion: {question}\nContext: {text}"
        answer_response = client.completions.create(
            model=engine,
            prompt=answer_prompt,
            max_tokens=a_max_tokens
        )
        answer = answer_response.choices[0].text.strip()

        return question, answer
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

# Function to prepare data for fine-tuning
def prepare_finetune_data(df, output_file_path):
    # Initialize a list to store formatted data
    formatted_texts = []

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        context = row['text']
        # Process each question-answer pair
        for i in range(1, 4):  # Assuming there are 3 pairs
            question_col = f'Generated Question {i}'
            answer_col = f'Generated Answer {i}'

            # Check if the answer is not null or empty
            if pd.notnull(row[answer_col]) and row[answer_col].strip():
                # Format the entry as per the LLaMA-2 authors' instructions
                formatted_text = f"<s>[INST] <<SYS>>\n{context}\n<</SYS>>\nQuestion: {row[question_col]}\n[/INST]\nHelpful Answer:\n{row[answer_col]}\n</s>"
                formatted_texts.append(formatted_text)

    # Create a new DataFrame with the 'formatted_text' column
    formatted_df = pd.DataFrame({'formatted_text': formatted_texts})

    # Save the formatted data to a new file
    formatted_df.to_csv(output_file_path, index=False)


def main():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = config.api_key

    # Check command line argument
    if len(sys.argv) < 2:
        print("Usage: python finetune-data-pipeline.py [generate|finetune]")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "generate":
        client = OpenAI()
        sheet_idx = 1  
        df = read_ods('./data/corporate/dataset_website-content-crawler-en.ods', sheet_idx)
        num_pairs_per_context = 3

        # Generate questions and answers
        for index, row in df.iterrows():
            text = row['text']
            for i in range(num_pairs_per_context):
                question, answer = generate_question_and_answer(client, text)
                if question and answer:
                    df.at[index, f'Generated Question {i+1}'] = question
                    df.at[index, f'Generated Answer {i+1}'] = answer

        # Save the updated dataframe
        df.to_csv('processed_data_multiple_qa.csv', index=False)

    elif mode == "finetune":
        # Load the data
        df = pd.read_csv('./data/corporate/processed_data_multiple_qa.csv')
        prepare_finetune_data(df, './data/corporate/finetune_data.csv')

    else:
        print("Invalid mode. Use 'generate' or 'finetune'.")

if __name__ == "__main__":
    main()
