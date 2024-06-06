# word embedding calculation
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined2.xlsx'
df = pd.read_excel(file_path)
column_name1 = 'emotion_id' 
column_name2 = 'predict9'
column_name4 = 'we9'
# Read the specified column
true_data = df[column_name1]
predict_data = df[column_name2]
word_similarity = df[column_name4]

def main():
    for i in range(min(6000, len(df))):  # Ensure we do not go out of bounds
        word1 = str(true_data[i])
        word2 = str(predict_data[i])
        similarity = calculate_cosine_similarity(word1, word2, glove_model)
        df.loc[i, column_name4] = similarity
        print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
    
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False)

def load_glove_model(file_path):
    print("Loading GloVe Model")
    glove_model = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print(f"Loaded {len(glove_model)} words.")
    return glove_model

# Path to the GloVe file
glove_file_path = '/Users/yangl/Downloads/glove/glove.6B.300d.txt'
glove_model = load_glove_model(glove_file_path)

def get_word_vector(word, model):
    if isinstance(word, str):
        return model.get(word.lower())
    return None

def calculate_cosine_similarity(word1, word2, model):
    vector1 = get_word_vector(word1, model)
    vector2 = get_word_vector(word2, model)
    if vector1 is not None and vector2 is not None:
        similarity = cosine_similarity([vector1], [vector2])
        return similarity[0][0]
    else:
        print(f"One of the words '{word1}' or '{word2}' does not exist in the model.")
        return None

if __name__ == "__main__":
    main()