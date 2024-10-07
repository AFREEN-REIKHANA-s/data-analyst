import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('C:\\Users\\surface\\Downloads\\task 3\\employee_review_mturk_dataset_v10_kaggle.csv')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Drop rows with missing feedback if necessary
df.dropna(subset=['feedback'], inplace=True)

# Perform basic sentiment analysis (positive/negative feedback)
def sentiment_analysis(feedback):
    positive_keywords = ['good', 'great', 'excellent', 'positive', 'satisfied']
    negative_keywords = ['bad', 'poor', 'negative', 'dissatisfied', 'worst']
    
    if any(word in feedback.lower() for word in positive_keywords):
        return 'Positive'
    elif any(word in feedback.lower() for word in negative_keywords):
        return 'Negative'
    else:
        return 'Neutral'

# Apply sentiment analysis
df['sentiment'] = df['feedback'].apply(sentiment_analysis)

# Count sentiment categories
sentiment_counts = df['sentiment'].value_counts()
print("\nSentiment counts:")
print(sentiment_counts)

# Plot sentiment distribution
plt.figure(figsize=(8, 5))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'grey'])
plt.title('Sentiment Distribution of Employee Feedback')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Identify common areas for improvement
# Assuming there is a 'comments' column for specific feedback
improvement_keywords = ['improve', 'need', 'lack', 'better', 'more', 'should']
df['areas_for_improvement'] = df['feedback'].apply(lambda x: any(word in x.lower() for word in improvement_keywords))

# Count areas for improvement
areas_counts = df['areas_for_improvement'].value_counts()
print("\nAreas for Improvement:")
print(areas_counts)

# Plot areas for improvement
plt.figure(figsize=(8, 5))
areas_counts.plot(kind='bar', color=['blue', 'orange'])
plt.title('Identified Areas for Improvement in Employee Feedback')
plt.xlabel('Areas for Improvement')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()