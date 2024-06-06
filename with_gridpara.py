# with grid para
import os
import openai
import pandas as pd
from sklearn.model_selection import ParameterGrid

# Load the dataset
file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined.xlsx'
df = pd.read_excel(file_path)
column_name1 = 'text'
column_name2 = 'predict1'
column_name3 = 'emotion_id'

# Set OpenAI API key
openai.api_key = "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"
model_name = "gpt-4o"

# Define the parameter grid
param_grid = {
    'temperature': [0.7, 1.0, 1.3],
    'max_tokens': [128, 256],
    'top_p': [0.9, 1.0],
    'frequency_penalty': [0, 0.5],
    'presence_penalty': [0, 0.5]
}

def main():
    best_params = None
    best_score = float('-inf')

    for params in ParameterGrid(param_grid):
        score = run_experiment(params)
        if score > best_score:
            best_score = score
            best_params = params

    print(f"Best parameters found: {best_params}")
    print(f"Best score: {best_score}")

    # Use best parameters to process the entire dataset
    process_data(best_params)

def run_experiment(params):
    scores = []
    for i in range(100):  # Use a subset for quick evaluation
        user_input = f'read the text "{df[column_name1][i]}" and return an adjective word that can best describe the emotion in the text. Only return 1 word!'
        response = chat_with_openai(user_input, params)
        # Assume a scoring function evaluate_response is defined
        score = evaluate_response(response, df['emotion_id'][i])  # Replace with actual scoring function
        scores.append(score)
    return sum(scores) / len(scores)

def evaluate_response(predicted, actual):
    # Simple evaluation: 1 if correct, 0 if incorrect
    return 1 if predicted.strip().lower() == actual.strip().lower() else 0

def chat_with_openai(prompt, params):
    messages = [
        {
            "role": "system",
            "content": "Read the information provided. Describe the emotion expressed in this information in one word. The word has to be an adjective and accurately indicate the emotion of the writer of the piece of information."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    try:
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            temperature=params['temperature'],
            max_tokens=params['max_tokens'],
            top_p=params['top_p'],
            frequency_penalty=params['frequency_penalty'],
            presence_penalty=params['presence_penalty']
        )
        return response.choices[0].message['content'].strip()
    
    except openai.error.InvalidRequestError as e:
        print(f"InvalidRequestError: {e}")
        return "InvalidRequestError occurred"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred"

def process_data(params):
    print("Processing full dataset with best parameters...")
    for i in range(len(df)):
        user_input = f'read the text "{df[column_name1][i]}" and return an adjective word that can best describe the emotion in the text. Only return 1 word!'
        response = chat_with_openai(user_input, params)
        df.loc[i, column_name2] = response
        print(f"Chatbot: {response}")

    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

if __name__ == "__main__":
    main()