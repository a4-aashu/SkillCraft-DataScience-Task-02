# SkillCraft Technology - Data Science Internship
# Task 02: Data Cleaning & Exploratory Data Analysis (EDA)
# Dataset: Titanic Dataset

# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load Dataset

df = pd.read_csv("titanic.csv")


# Display first rows

print(df.head())


# Dataset information

print(df.info())


# Check missing values

print(df.isnull().sum())


# Data Cleaning

# Fill missing Age values with mean age
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked values
df['Embarked'].fillna(
    df['Embarked'].mode()[0],
    inplace=True
)


# Drop unnecessary column

df.drop(
    columns=['Cabin'],
    inplace=True
)


# Summary Statistics

print(df.describe())


# Exploratory Data Analysis


# Survival Count

sns.countplot(
    x='Survived',
    data=df
)

plt.title("Survival Count")
plt.show()


# Gender vs Survival

sns.countplot(
    x='Sex',
    hue='Survived',
    data=df
)

plt.title("Gender Survival Analysis")
plt.show()


# Passenger Class Analysis

sns.countplot(
    x='Pclass',
    hue='Survived',
    data=df
)

plt.title("Passenger Class vs Survival")
plt.show()


# Age Distribution

sns.histplot(
    df['Age'],
    bins=30
)

plt.title("Age Distribution")
plt.show()


print("EDA Completed Successfully")
