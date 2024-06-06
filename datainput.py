#sentient closeness
import os

# Import the openai package
import openai
import pandas as pd
import numpy as np
from textblob import TextBlob

file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined2.xlsx'
df = pd.read_excel(file_path)
column_name1 = 'emotion_id'  # Change this to your column name
column_name2 = 'predict9'
column_name3 = 'sc9'

# Read the specified column
true_data = df[column_name1]
predict_data = df[column_name2]
sentiment_closeness = df[column_name3]

def get_sentiment_polarity(word):
    if isinstance(word, str):
        return TextBlob(word).sentiment.polarity
    return 0.0

def main():
    for i in range(min(6000, len(df))):  # Ensure we do not go out of bounds
        word1 = true_data[i]
        word2 = predict_data[i]

        # Get sentiment polarity for each word
        word1_sentiment = get_sentiment_polarity(word1)
        word2_sentiment = get_sentiment_polarity(word2)
        
        # Calculate the closeness (e.g., absolute difference)
        closeness = 1 - abs(word1_sentiment - word2_sentiment)
        df.loc[i, column_name3] = closeness
        print(f"Sentiment closeness of {word1} and {word2} is: {closeness}")
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

if __name__ == "__main__":
    main()

