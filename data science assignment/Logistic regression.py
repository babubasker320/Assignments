import pandas as pd

# Load the Titanic datasets
train_data_path = "D:\Excelr\Data Science\Data Science Assignment\Logistic Regression\Logistic Regression\Titanic_train.csv"
test_data_path = "D:\Excelr\Data Science\Data Science Assignment\Logistic Regression\Logistic Regression\Titanic_test.csv"
# Load the data
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)
# Display the first few rows
train_df_info = train_df.info()
train_df_head = train_df.head()
# Generate summary statistics for numerical features
train_df_describe = train_df.describe()

train_df_info, train_df_head, train_df_describe

import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")
# 1. Histogram of Age
plt.figure(figsize=(8, 6))
sns.histplot(train_df['Age'].dropna(), bins=30, kde=True, color='blue')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 2. Box plot of Fare by Passenger Class (Pclass)
plt.figure(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=train_df)
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# 3. Pair plot to visualize relationships between numerical features
numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
sns.pairplot(train_df[numerical_features], hue='Survived', diag_kind='kde', palette="Set2")
plt.show()

# Handle missing values by removing rows with NaNs
clean_train_df = train_df[numerical_features].dropna()

# the pair plot with the cleaned data
sns.pairplot(clean_train_df, hue='Survived', diag_kind='kde', palette="Set2")
plt.show()

# Generate a correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

# Scatter plot of Age vs. Fare colored by Survival
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=train_df, palette="Set1")
plt.title('Age vs. Fare with Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# 1. Handle missing values
# Impute missing 'Age' with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Drop the 'Cabin' column due to too many missing values
train_df.drop('Cabin', axis=1, inplace=True)

# Impute missing 'Embarked' values with the most frequent value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# 2. Encode categorical variables
# Encode 'Sex' as 0 for male and 1 for female
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})

# Encode 'Embarked' using one-hot encoding
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)

# Check the processed data
train_df.head()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Separate features (X) and target (y)
X = train_df.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket'])  # Exclude irrelevant columns
y = train_df['Survived']

# 2. Split the data into training and validation sets (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the logistic regression model
log_reg_model = LogisticRegression(max_iter=1000)

# 4. Train the model using the training data
log_reg_model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = log_reg_model.predict(X_val)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_val, y_pred)

accuracy

import pickle

# Save the trained model
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(log_reg_model, f)


from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Calculate precision, recall, F1-score, and ROC-AUC score
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred)

# Get the ROC curve data
fpr, tpr, thresholds = roc_curve(y_val, log_reg_model.predict_proba(X_val)[:, 1])

# Visualize the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# the computed metrics
precision, recall, f1, roc_auc

# Extract the coefficients of the logistic regression model
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_reg_model.coef_[0]
})

# Sort by the absolute value of the coefficients to see the most impactful features
coefficients_sorted = coefficients.sort_values(by='Coefficient', ascending=False)

coefficients_sorted