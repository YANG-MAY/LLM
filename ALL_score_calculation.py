#4 metrics score calculation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors

# Load the dataset


file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined2.xlsx'
column_name5 = 'text'
column_name6 = 'predict9'
column_name9 = 'emotion_id'
df = pd.read_excel(file_path)
df = df.dropna(subset=[column_name5, column_name6, column_name9])
test_data = df[column_name5]
true_data = df[column_name9]
predict_data = df[column_name6]
print(test_data)

true_labels = df['emotion_id'].astype(str).str.lower()
predicted_labels = df['predict9'].astype(str).str.lower()
   
detected_emotions = np.array(predicted_labels)
ground_truth_emotions = np.array(true_labels)
ground_truth_emotions = ground_truth_emotions.astype(str)
detected_emotions = detected_emotions.astype(str)
accuracy = accuracy_score(ground_truth_emotions, detected_emotions)
precision = precision_score(ground_truth_emotions, detected_emotions, average='macro', zero_division=0)
recall = recall_score(ground_truth_emotions, detected_emotions, average='macro', zero_division=0)
f1 = f1_score(ground_truth_emotions, detected_emotions, average='macro', zero_division=0)

print(f'test Accuracy: {accuracy:.2f}')
print(f'test Precision: {precision:.2f}')
print(f'test Recall: {recall:.2f}')
print(f'test F1 Score: {f1:.2f}')    



cm = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

# Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index=np.unique(true_labels), columns=np.unique(true_labels))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()