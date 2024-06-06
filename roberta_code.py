#roberta code
# Load model directly

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd

file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined2.xlsx'
df = pd.read_excel(file_path)
column_name1 = 'text'  # Change this to your column name
column_name2 = 'predict5'
# Read the specified column
column_data = df[column_name1]
predict_data = df[column_name2]

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-emotion-multilabel-latest")
# Use a pipeline as a high-level helper
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1, truncation=True)

# Get the prediction for a sample text

def main():
    for i in range(len(df)):  # Iterate through the DataFrame rows
        text = str(column_data[i])  # Ensure the text is a string
        result = pipe(text)  # Pass the text directly
        response = result[0][0]['label']  # Access the first dictionary in the list and get the label
        df.loc[i, column_name2] = response  # Store the response in the specified column
        print(f"Predicted Emotion: {response}\n")

    # Save all changes after the loop completes
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)
        


if __name__ == "__main__":
    main() 