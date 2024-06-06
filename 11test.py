# Labbel method of 
# This file is used to perfom the detection without tarin, text, 
import os

# Import the openai package
import openai
import pandas as pd

file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined2.xlsx'
df = pd.read_excel(file_path)
column_name1 = 'text'  # Change this to your column name
column_name2 = 'predict4'
# Read the specified column
column_data = df[column_name1]
predict_data = df[column_name2]

openai.api_key = "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"
model_name="gpt-4o"

def main():
   
    print("Welcome to Chatbot! The emotion detected is:")
    for i in range(6000):  # Ensure you have at least 10 rows in df to avoid IndexError
        user_input = f'read the text "{column_data[i]}" and choose a word in the list:"Anticipation, Love, Optimism, Pessimism, Trust, Happiness, Surprise, Disgust, Fear, Sadness, Anger" to describe the emotion of the text. Only return the selected one word!'
        # predict2 list used for 'read the text "{column_data[i]}" and choose a emotion word in list : Annoyed, Nervous, Lonely, Furious, Confused, Ashamed, Angry, Proud, Sad, Shocked, Frustrated, Embarrassed, Surprised, Amazed, Amused, Anxious, Exhausted, Curious, Awkward, Happy, Disappointed, Disgusted, Excited, Stressed. to describe the emotion of the text.
        # list used for predict3 : Stressed, Excited, Disgusted, Disappointed, Happy, Awkward, Curious, Exhausted, Anxious, Amused, Amazed, Surprised, Embarrassed, Frustrated, Shocked, Sad, Proud, Angry, Ashamed, Confused,  Furious, Lonely, Nervous, Annoyed.
        # predict 4: read the text "{column_data[i]}" and choose a word in the list:"Anger, Sadness, Fear, Feelings, Surprise, Happiness" to describe the emotion of the text. Only return the selected one word!'
        # prdict 5 read the text "{column_data[i]}" and choose a word in the list:"Anger, Sadness, Fear, Feelings, Surprise, Happiness" to describe the emotion of the text. Only return the selected one word! Do not return any word that is not in the list "Anger, Sadness, Fear, Feelings, Surprise, Happiness"
        import time
        time.sleep(0.1)
        response = chat_with_openai(user_input)  # Pass user_input as an argument
        df.loc[i, column_name2] = response
        print(f"Chatbot: {response}")

    # Save all changes after the loop completes
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

def chat_with_openai(prompt):
    messages = [
        {
            "role": "system",
            "content": "Read the information provided.select the emotion expressed in this information in one word. The word has to be an noun and accurately indicate the emotion of the writer of the piece of text."
        # Read the information provided. Describe the emotion expressed in this information in one word. The word has to be an adjective and accurately indicate the emotion of the writer of the piece of information
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
            temperature=1.0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message['content']
    
    except openai.error.InvalidRequestError as e:
        print(f"InvalidRequestError: {e}")
        return "InvalidRequestError occurred"
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred"
    
    # Extract the chatbot's message from the response.
    # Assuming there's at least one response and taking the last one as the chatbot's reply.
    
    chatbot_response = response['choices'][0]['message']['content']
    return chatbot_response.strip()


if __name__ == "__main__":
    main() 


#from openai import OpenAI
#client = openai()

#completion = client.chat.completions.create(
#  model="gpt-3.5-turbo",
#  messages=[
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": "Hello!"}
#  ]
#)

#print(completion.choices[0].message)
