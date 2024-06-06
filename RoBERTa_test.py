from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = '/Users/yangl/Desktop/Dataset/vent/new_data_refined.xlsx'
column_name1 = 'text'  # Text column
column_name2 = 'predictR0'  # Prediction column
column_name3 = 'emotion_cat'  # True label column

df = pd.read_excel(file_path)
df = df.dropna(subset=[column_name1, column_name2, column_name3])

true_labels = df[column_name3].astype(str)
predicted_labels = df[column_name2].astype(str)

# Ensure there are no discrepancies in the labels
print("Unique true labels:", true_labels.unique())
print("Unique predicted labels:", predicted_labels.unique())

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

print(f'test Accuracy: {accuracy:.2f}')
print(f'test Precision: {precision:.2f}')
print(f'test Recall: {recall:.2f}')
print(f'test F1 Score: {f1:.2f}')

# Generate confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=np.unique(true_labels))

# Create a DataFrame for better visualization
cm_df = pd.DataFrame(cm, index = np.unique(true_labels), columns=np.unique(true_labels))

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', xticklabels=cm_df.columns, yticklabels=cm_df.index)
plt.title('Confusion Matrix')
plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()