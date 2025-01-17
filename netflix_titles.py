# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:00:24 2025

@author: Dell
"""

# Import required libraries for data manipulation, visualization, and word cloud generation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset (replace with the correct file path)
# **Business Objective:** Load the dataset to perform analysis and build a recommendation system
# **Impact:** Properly loading the dataset ensures we have the data available for further processing and analysis.
df = pd.read_csv('E:/Honars(DS)/Data Science/Netflix Content Recommendation System/netflix_titles.csv')

# Step 1: Data Preprocessing

# Check for missing values in the dataset
# **Business Objective:** Identify missing data that may hinder analysis or model training.
# **Impact:** Handling missing values ensures the data is clean and reliable for building a recommendation system.
print("Missing values in each column:")
print(df.isnull().sum())

# Remove rows with missing 'title' or 'description'
# **Business Objective:** Remove incomplete rows to ensure that we are only analyzing valid records.
# **Impact:** Reduces noise in the dataset and ensures that the recommendation system has complete data to work with.
df.dropna(subset=['title', 'description'], inplace=True)

# Check for duplicate rows in the dataset
# **Business Objective:** Identify duplicate rows that might distort the analysis or model results.
# **Impact:** Removing duplicates ensures that each movie/TV show is considered only once, leading to a fairer and more accurate recommendation.
print("\nDuplicate rows in the dataset:")
print(df.duplicated().sum())

# Remove duplicate rows based on the 'title' column
# **Business Objective:** Remove duplicate entries to ensure unique data points for building recommendations.
# **Impact:** Guarantees that duplicate records don’t skew the analysis or recommendations.
df.drop_duplicates(subset=['title'], inplace=True)

# Clean 'description' text - remove non-alphabetic characters, lowercase
# **Business Objective:** Standardize the textual data for better analysis and feature extraction.
# **Impact:** Ensures that the text data is uniform and free from irrelevant characters, allowing accurate analysis for content-based recommendations.
df['cleaned_description'] = df['description'].str.replace(r'[^a-zA-Z\s]', '', regex=True).str.lower()

# Step 2: Exploratory Data Analysis (EDA)

# 2.1: Display basic information about the dataset
# **Business Objective:** Understand the structure and types of data in the dataset before performing any analysis.
# **Impact:** Helps in identifying which columns need preprocessing or transformation, and validates the dataset’s integrity.
print("\nBasic Information about the dataset:")
print(df.info())

# 2.2: Summary statistics for numerical columns
# **Business Objective:** Gain insights into the distribution and central tendencies of numerical features like 'duration' or 'rating'.
# **Impact:** Provides an overview of the dataset’s numerical attributes, helping identify outliers, missing values, or skewness that might affect model performance.
print("\nSummary statistics for numerical columns:")
print(df.describe())

# 2.3: Distribution of data - Number of Movies/TV Shows by Genre
# **Business Objective:** Analyze the distribution of movies/TV shows by genre to understand content diversity.
# **Impact:** Helps identify which genres are most prevalent, guiding content recommendations based on genre popularity.
plt.figure(figsize=(10, 6))
genre_counts = df['genre'].value_counts().head(10)  # Top 10 genres
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis")
plt.title('Top 10 Genres')
plt.xlabel('Number of Movies/TV Shows')
plt.ylabel('Genre')
plt.show()

# 2.4: WordCloud for most frequent words in the descriptions
# **Business Objective:** Visualize the most common terms in movie/TV show descriptions.
# **Impact:** WordClouds can provide insights into recurring themes or keywords, which could enhance content-based recommendation algorithms.
all_descriptions = ' '.join(df['cleaned_description'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_descriptions)

# Display the WordCloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# 2.5: Distribution of movies/TV shows by release year
# **Business Objective:** Examine how movies and TV shows are distributed over time to understand trends in content production.
# **Impact:** Helps in identifying trends and tailoring recommendations based on the release year, which could be useful for creating recommendations based on movie/TV show age.
plt.figure(figsize=(10, 6))
df['release_year'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Number of Movies/TV Shows by Release Year')
plt.xlabel('Release Year')
plt.ylabel('Count')
plt.show()

# 2.6: Relationship between Ratings and Release Year (if available)
# **Business Objective:** Analyze how ratings change over time to identify if more recent movies/TV shows are better rated.
# **Impact:** Understanding ratings over time may inform the recommendation system to prioritize newer or higher-rated content.
if 'rating' in df.columns and 'release_year' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='release_year', y='rating', data=df)
    plt.title('Distribution of Ratings by Release Year')
    plt.xlabel('Release Year')
    plt.ylabel('Rating')
    plt.xticks(rotation=90)
    plt.show()

# 2.7: Plot distribution of movie durations (if available)
# **Business Objective:** Explore the duration of movies/TV shows to understand preferences for content length.
# **Impact:** Duration could be a key factor in recommendations, especially if users prefer shorter or longer movies/shows.
if 'duration' in df.columns:
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')  # Convert to numeric, coerce errors
    plt.figure(figsize=(10, 6))
    df['duration'].dropna().plot(kind='hist', bins=30, color='orange', edgecolor='black')
    plt.title('Distribution of Movie Duration')
    plt.xlabel('Duration (in minutes)')
    plt.ylabel('Frequency')
    plt.show()

# 2.8: Show the first few rows of the dataset to verify cleaning
# **Business Objective:** Preview the cleaned dataset to ensure that the preprocessing steps were successful.
# **Impact:** Helps confirm that the data is properly cleaned and ready for analysis and modeling.
print("\nFirst few rows of the cleaned dataset:")
print(df.head())
