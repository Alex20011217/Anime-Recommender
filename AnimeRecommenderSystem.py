from nltk.stem import WordNetLemmatizer
from tqdm import tqdm 
import nltk
import re
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

import warnings

nltk.download('punkt')

warnings.filterwarnings("ignore")

data = pd.read_csv('AnimeWorld.csv')
anime_data = data.iloc[0:500, :]

data.head(10)

anime_data = anime_data[['Anime','Genre','Description','Studio']]
anime_data.sample(5)

sns.heatmap(anime_data.isnull())

for col in anime_data.columns:
    empty = anime_data[col].isnull().sum()
    percent = empty/len(anime_data)*100
    print(f"{col}:{percent}% empty data")
print("========================================")
anime_data.info(0)

def WC_generate(col,size,words):
    plt.figure(figsize=(15,15))
    wordcloud = WordCloud(stopwords=STOPWORDS,background_color = 'black', width = size,  height = size, max_words = words)
    wordcloud.generate(' '.join(anime_data[col].astype(str)))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(f"{col} Wordcloud",fontsize = 24)
    plt.show()

WC_generate('Anime',1050,150)

WC_generate('Studio',1000,100)

tqdm.pandas()

def clean_text(text):
    
    # Remove symbols like Â° and non-ASCII characters
    text = ''.join(char for char in text if ord(char) < 128)
    
    # Remove special characters and numbers using regular expressions
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Initialize a WordNet lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Apply lemmatization to each token and remove stopwords
    cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]
    
    # Remove blank tokens
    cleaned_tokens = [token for token in cleaned_tokens if token.strip()]
    
    # Join the cleaned tokens to form a single string
    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text

def filter_genre(data):
    if data[0]=='[':
        return data.strip('[]').replace(' ','').replace("'",'')
    else:
        return data

# Apply the filter_genre function to 'Genre' column 
anime_data['cleaned_Genre'] = anime_data['Genre'].progress_apply(filter_genre)
print("Original 'Genre' column:")
print(anime_data['Genre'].head())

print("\nFiltered 'Genre' column:")
print(anime_data['cleaned_Genre'].head())
    
# Apply the clean_text function to 'Anime' column with a progress bar
anime_data['cleaned_Anime'] = anime_data['Anime'].progress_apply(clean_text)

print("Original 'Anime' column:")
print(anime_data['Anime'].head())

print("\nCleaned 'Anime' column:")
print(anime_data['cleaned_Anime'].head())

# Apply the clean_text function to 'Description' column with a progress bar
anime_data['cleaned_Description'] = anime_data['Description'].progress_apply(clean_text)

print("Original 'Description' column:")
print(anime_data['Description'].head())

print("\nCleaned 'Description' column:")
print(anime_data['cleaned_Description'].head())

WC_generate('cleaned_Genre',500,50)

WC_generate('cleaned_Anime',1050,150)

WC_generate('cleaned_Description',1050,150)

from fuzzywuzzy import fuzz

# Function to find the closest matching anime title
def find_closest_match(user_input, anime_titles):
    cleaned_input = clean_text(user_input)
    best_match = None
    best_score = -1  # Initialize with a score that ensures the first match will be considered
    for title in anime_titles:
        #similarity_score = fuzz.ratio(user_input.lower(), title.lower())
        similarity_score = fuzz.token_set_ratio(user_input.lower(), title.lower())
        if similarity_score > best_score:
            best_match = title
            best_score = similarity_score
    return best_match

#TFIDF

# Create the TF-IDF vectorizer
desc_tfidf = TfidfVectorizer(stop_words='english')

# Fit the vectorizer with your data
desc_tfidf_matrix = desc_tfidf.fit_transform(anime_data['cleaned_Description'])

print("===========================")
print("  Description Features     ")
print("===========================\n")
print(desc_tfidf.get_feature_names_out()[200:400])
print("\nSize of features name: " + str(len(desc_tfidf.get_feature_names_out()))) 
print("Matrix shape: " + str(desc_tfidf_matrix.shape))
desc_tfidf_matrix

# Create the TF-IDF vectorizer
genre_tfidf = TfidfVectorizer(stop_words='english')

# Fit the vectorizer with your data
genre_tfidf_matrix = genre_tfidf.fit_transform(anime_data['cleaned_Genre'])

print ("===========================")
print ("       Genre  Features     ")
print ("===========================\n")
print (genre_tfidf.get_feature_names_out())
print ("\nSize of features name: " + str(len(genre_tfidf.get_feature_names_out()))) 
print ("Matrix shape: " + str(genre_tfidf_matrix.shape))
genre_tfidf_matrix

#TFIDF + Cosine

TC_Anime_desc_cosine_sim = cosine_similarity(desc_tfidf_matrix,desc_tfidf_matrix)
TC_Anime_desc_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()
print (TC_Anime_desc_cosine_sim.shape)
print("==========================================================================")
print(TC_Anime_desc_cosine_sim)
print("==========================================================================")
print(TC_Anime_desc_index_sim.shape)
print("==========================================================================")
print(TC_Anime_desc_index_sim.head())

TC_Anime_genre_cosine_sim = cosine_similarity(genre_tfidf_matrix,genre_tfidf_matrix)
TC_Anime_genre_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()
print (TC_Anime_genre_cosine_sim.shape)
print("==========================================================================")
print(TC_Anime_genre_cosine_sim)
print("==========================================================================")
print(TC_Anime_genre_index_sim.shape)
print("==========================================================================")
print(TC_Anime_genre_index_sim)

def TFIDF_Cosine_get_recommendations1(user_input):
    closest_match = find_closest_match(user_input, anime_data['cleaned_Anime'])
    idx = TC_Anime_desc_index_sim[closest_match]
    sim_scores = list(enumerate(TC_Anime_desc_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    print("Closest match: "+ closest_match)
    VerOne_Reccomendations = anime_data.iloc[anime_indices]
    
    return TFIDF_Cosine_get_recommendations2(VerOne_Reccomendations, closest_match)

def TFIDF_Cosine_get_recommendations2(recommendations, closest_match):
    # Check if there are any recommendations
    if len(recommendations) == 0:
        return "No recommendations available."
    
    anime_title = closest_match
    
    # Get the index of the closest match in the desc_index_sim Series
    idx = TC_Anime_genre_index_sim[anime_title]
    
    # Sort recommendations by cosine similarity to the selected user input
    recommendations['Similarity_Score'] = TC_Anime_genre_cosine_sim[0]
    recommendations = recommendations.sort_values(by='Similarity_Score', ascending=False)
    
    # Get the top N refined recommendations (adjust N as needed)
    top_recommendations = recommendations[0:11]
    print('Anime title: ' + anime_title)
    
    return top_recommendations[['Anime', 'cleaned_Genre', 'Description']]

#Final version of anime recommendations
user_input = input('Please enter one fo the favorite anime name for more recommendations: ')
RefinedVer_Recommendations = TFIDF_Cosine_get_recommendations1(user_input)
pd.DataFrame(RefinedVer_Recommendations[['Anime', 'cleaned_Genre', 'Description']][0:11])

#TFIDF + Pearson
from scipy.stats import pearsonr
from tqdm import tqdm  # Import tqdm for progress bar

# Define a function to calculate correlations for a single row
def calculate_correlations(i):
    TP_corr_row = []
    for j in range(len(desc_tfidf_matrix.toarray())):
        corr, _ = pearsonr(
            desc_tfidf_matrix[i].toarray().flatten(),
            desc_tfidf_matrix[j].toarray().flatten()
        )
        TP_corr_row.append(corr)
    return TP_corr_row

# Define a function to calculate Pearson correlations sequentially
def desc_calculate_pearson_correlations_sequential(desc_tfidf_matrix):
    # Initialize a list to store Pearson correlations
    TP_Anime_desc_pearson_corr = []

    # Use tqdm for a progress bar
    for i in tqdm(range(len(desc_tfidf_matrix.toarray()))):
        TP_corr_row = calculate_correlations(i)
        TP_Anime_desc_pearson_corr.append(TP_corr_row)

    # Create a DataFrame to hold the correlation values
    TP_Anime_desc_pearson_sim = pd.DataFrame(TP_Anime_desc_pearson_corr)

    # Create a Series to map anime titles to their indices
    TP_Anime_desc_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

    return TP_Anime_desc_pearson_sim, TP_Anime_desc_index_sim

def genre_calculate_pearson_correlations_sequential(genre_tfidf_matrix):
    # Initialize a list to store Pearson correlations for genre
    TP_Anime_genre_pearson_corr = []

    # Use tqdm for a progress bar
    for i in tqdm(range(len(genre_tfidf_matrix.toarray()))):
        TP_corr_row = calculate_correlations(i)
        TP_Anime_genre_pearson_corr.append(TP_corr_row)

    # Create a DataFrame to hold the correlation values for genre
    TP_Anime_genre_pearson_sim = pd.DataFrame(TP_Anime_genre_pearson_corr)

    # Create a Series to map anime titles to their indices for genre
    TP_Anime_genre_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Genre']).drop_duplicates()

    return TP_Anime_genre_pearson_sim, TP_Anime_genre_index_sim


# Call the function to calculate Pearson correlations sequentially
TP_Anime_desc_pearson_sim, TP_Anime_desc_index_sim = desc_calculate_pearson_correlations_sequential(desc_tfidf_matrix)

# Print or use the results as needed
print("Shape of Anime_desc_pearson_sim: " + str(TP_Anime_desc_pearson_sim.shape))
#print("==========================================================================")
#print(Anime_desc_pearson_sim)
print("==========================================================================")
print("Shape of Anime_desc_index_sim: " + str(TP_Anime_desc_index_sim.shape))
#print("==========================================================================")
#print(Anime_desc_index_sim.head())

# Initialize a list to store Pearson correlations
TP_Anime_genre_pearson_corr = []

def calculate_pearson_correlations_sequential(genre_tfidf_matrix):
    # Initialize a list to store Pearson correlations
    TP_Anime_genre_pearson_corr = []

    # Use tqdm for a progress bar
    for i in tqdm(range(len(genre_tfidf_matrix.toarray()))):
        TP_corr_row = calculate_correlations(i)
        TP_Anime_genre_pearson_corr.append(TP_corr_row)

    # Create a DataFrame to hold the correlation values
    TP_Anime_genre_pearson_sim = pd.DataFrame(TP_Anime_genre_pearson_corr)

    # Create a Series to map anime titles to their indices
    TP_Anime_genre_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Genre']).drop_duplicates()

    return TP_Anime_genre_pearson_sim, TP_Anime_genre_index_sim

# Call the function to calculate Pearson correlations sequentially
TP_Anime_genre_pearson_sim, TP_Anime_genre_index_sim = genre_calculate_pearson_correlations_sequential(genre_tfidf_matrix)

# Print or use the results as needed
print("Shape of Anime_desc_pearson_sim: " + str(TP_Anime_desc_pearson_sim.shape))
print("==========================================================================")
#print(Anime_genre_pearson_sim)
#print("==========================================================================")
print("Shape of Anime_genre_index_sim: " + str(TP_Anime_genre_index_sim.shape))
#print("==========================================================================")
#print(Anime_genre_index_sim)

def TFIDF_Pearson_get_recommendations1(user_input):
    closest_match = find_closest_match(user_input, anime_data['cleaned_Anime'])
    idx = TP_Anime_desc_index_sim[closest_match]
    
    # Calculate Pearson correlation coefficients
    TP_pearson_scores = []
    for i in range(len(TP_Anime_desc_pearson_sim)):
        TP_corr, _ = pearsonr(TP_Anime_desc_pearson_sim[idx], TP_Anime_desc_pearson_sim[i])
        TP_pearson_scores.append((i, TP_corr))
    
    # Sort scores by correlation coefficient
    TP_pearson_scores = sorted(TP_pearson_scores, key=lambda x: x[1], reverse=True)
    
    anime_indices = [i[0] for i in TP_pearson_scores]
    print("Closest match: " + closest_match)
    Pearson_Recommendations = anime_data.iloc[anime_indices]
    
    print(pd.DataFrame(Pearson_Recommendations[['cleaned_Anime', 'cleaned_Genre', 'cleaned_Description']]))
    
    return TFIDF_Pearson_get_recommendations2(Pearson_Recommendations, closest_match)

def TFIDF_Pearson_get_recommendations2(recommendations, closest_match):
    # Check if there are any recommendations
    if len(recommendations) == 0:
        return "No recommendations available."

    idx = 0
    
    TP_Anime_genre_pearson_sim_df = pd.DataFrame(TP_Anime_genre_pearson_sim, index=anime_data['cleaned_Anime'], columns=anime_data['cleaned_Anime'])
    
    TP_sim_scores = list(enumerate(TP_Anime_genre_pearson_sim_df.iloc[idx]))
    TP_sim_scores = sorted(TP_sim_scores, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in TP_sim_scores]
    
    #anime_indices = [i[0] for i in pearson_scores]
    print('Anime title: ' + closest_match)
    
    return recommendations.iloc[anime_indices[0:11]]

# Example of using Pearson correlation-based recommendations
user_input = input('Please enter one fo the favorite anime name for more recommendations: ')
Pearson_Recommendations = TFIDF_Pearson_get_recommendations1(user_input)
pd.DataFrame(Pearson_Recommendations[['Anime', 'cleaned_Genre', 'Description']][0:4])

#BOW
from sklearn.feature_extraction.text import CountVectorizer

# Create the CountVectorizer
desc_count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer with your data
desc_count_matrix = desc_count_vectorizer.fit_transform(anime_data['cleaned_Description'])

print("===========================")
print("  Description Features     ")
print("===========================\n")
print(desc_count_vectorizer.get_feature_names_out()[200:400])
print("\nSize of features name: " + str(len(desc_count_vectorizer.get_feature_names_out()))) 
print("Matrix shape: " + str(desc_count_matrix.shape))
desc_count_matrix

# Create the CountVectorizer
genre_count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer with your data
genre_count_matrix = genre_count_vectorizer.fit_transform(anime_data['cleaned_Genre'])

print("===========================")
print("    Genre Features         ")
print("===========================\n")
print(genre_count_vectorizer.get_feature_names_out())
print("\nSize of features name: " + str(len(genre_count_vectorizer.get_feature_names_out()))) 
print("Matrix shape: " + str(genre_count_matrix.shape))
genre_count_matrix

#BOW + Cosine

# Create the CountVectorizer
desc_count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer with your description data
desc_count_matrix = desc_count_vectorizer.fit_transform(anime_data['cleaned_Description'])

# Calculate cosine similarity using the BOW representation
BC_Anime_desc_cosine_sim = cosine_similarity(desc_count_matrix, desc_count_matrix)

# Create a Series to map anime titles to their indices
BC_Anime_desc_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

print("Shape of Anime_desc_cosine_sim: " + str(BC_Anime_desc_cosine_sim.shape))
print("==========================================================================")
print(BC_Anime_desc_cosine_sim)
print("==========================================================================")
print("Shape of Anime_desc_index_sim: " + str(BC_Anime_desc_index_sim.shape))
print("==========================================================================")
print(BC_Anime_desc_index_sim.head())

# Create the CountVectorizer
genre_count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer with your genre data
genre_count_matrix = genre_count_vectorizer.fit_transform(anime_data['cleaned_Genre'])

# Calculate cosine similarity using the BOW representation
BC_Anime_genre_cosine_sim = cosine_similarity(genre_count_matrix, genre_count_matrix)

# Create a Series to map anime titles to their indices
BC_Anime_genre_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

print("Shape of Anime_genre_cosine_sim: " + str(BC_Anime_genre_cosine_sim.shape))
print("==========================================================================")
print(BC_Anime_genre_cosine_sim)
print("==========================================================================")
print("Shape of Anime_genre_index_sim: " + str(BC_Anime_genre_index_sim.shape))
print("==========================================================================")
print(BC_Anime_genre_index_sim)

def BOW_Cosine_get_recommendations1(user_input):
    closest_match = find_closest_match(user_input, anime_data['cleaned_Anime'])
    idx = BC_Anime_desc_index_sim[closest_match]
    sim_scores = list(enumerate(BC_Anime_desc_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    #sim_scores = sim_scores[1:11]
    anime_indices = [i[0] for i in sim_scores]
    print("Closest match: " + closest_match)
    VerOne_Reccomendations = anime_data.iloc[anime_indices]
    
    # VerOne_Reccomendations_Print =
    print(pd.DataFrame(VerOne_Reccomendations[['cleaned_Anime', 'cleaned_Genre', 'cleaned_Description']]))
    # print (VerOne_Reccomendations_Print)
    
    return BOW_Cosine_get_recommendations2(VerOne_Reccomendations, closest_match)

def BOW_Cosine_get_recommendations2(recommendations, closest_match):
    # Check if there are any recommendations
    if len(recommendations) == 0:
        return "No recommendations available."
    
    anime_title = closest_match
    
    # Get the index of the closest match in the desc_index_sim Series
    idx = BC_Anime_genre_index_sim[anime_title]
    
    # Sort recommendations by cosine similarity to the selected user input
    recommendations['Similarity_Score'] = BC_Anime_genre_cosine_sim[idx]
    recommendations = recommendations.sort_values(by='Similarity_Score', ascending=False)
    
    # Get the top N refined recommendations (adjust N as needed)
    top_recommendations = recommendations[0:11]
    print('Anime title: ' + anime_title)
    
    return top_recommendations

# Calculate BOW representation and cosine similarity for descriptions
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

desc_count_vectorizer = CountVectorizer(stop_words='english')
desc_count_matrix = desc_count_vectorizer.fit_transform(anime_data['cleaned_Description'])
Anime_desc_cosine_sim = cosine_similarity(desc_count_matrix, desc_count_matrix)

# Create a Series to map anime titles to their indices
Anime_desc_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

# Final version of anime recommendations using BOW and cosine similarity
user_input = input('Please enter one fo the favorite anime name for more recommendations: ')
RefinedVer_Recommendations = BOW_Cosine_get_recommendations1(user_input)
pd.DataFrame(RefinedVer_Recommendations[['Anime', 'cleaned_Genre', 'cleaned_Description']])

#BOW + Pearson

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from tqdm import tqdm  # Import tqdm for progress bar

# Create the CountVectorizer
desc_count_vectorizer = CountVectorizer(stop_words='english')

# Fit the vectorizer with your description data
desc_count_matrix = desc_count_vectorizer.fit_transform(anime_data['cleaned_Description'])

# Calculate Pearson correlation using the BOW representation
BC_Anime_desc_pearson_sim = np.zeros((len(anime_data), len(anime_data)))

# Iterate through each pair of anime titles
for i in tqdm(range(len(anime_data)), desc="Calculating Pearson Correlations"):
    for j in range(len(anime_data)):
        if i != j:  # Exclude self-comparisons
            corr, _ = pearsonr(
                desc_count_matrix[i].toarray().flatten(),
                desc_count_matrix[j].toarray().flatten()
            )
            BC_Anime_desc_pearson_sim[i, j] = corr

# Create a DataFrame to hold the correlation values
BC_Anime_desc_pearson_sim_df = pd.DataFrame(BC_Anime_desc_pearson_sim, index=anime_data['cleaned_Anime'], columns=anime_data['cleaned_Anime'])

# Create a Series to map anime titles to their indices
BC_Anime_desc_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

print("Shape of Anime_desc_pearson_sim: " + str(BC_Anime_desc_pearson_sim_df.shape))
print("==========================================================================")
print(BC_Anime_desc_pearson_sim_df)
print("==========================================================================")
print("Shape of Anime_desc_index_sim: " + str(BC_Anime_desc_index_sim.shape))
print("==========================================================================")
print(BC_Anime_desc_index_sim.head())

BC_Anime_genre_pearson_sim = None  # Initialize it in the outer block

# Define a function to calculate Anime_genre_pearson_sim
def calculate_genre_pearson_sim(anime_data):
    # Create the CountVectorizer for genre data
    genre_count_vectorizer = CountVectorizer(stop_words='english')

    # Fit the vectorizer with your genre data
    genre_count_matrix = genre_count_vectorizer.fit_transform(anime_data['cleaned_Genre'])

    # Calculate Pearson correlation using the BOW representation
    BC_Anime_genre_pearson_sim = np.zeros((len(anime_data), len(anime_data)))

    for i in tqdm(range(len(anime_data)), desc="Calculating Pearson Correlations for Genre"):
        for j in range(len(anime_data)):
            if i != j:  # Exclude self-comparisons
                BC_corr, _ = pearsonr(
                    genre_count_matrix[i].toarray().flatten(),
                    genre_count_matrix[j].toarray().flatten()
                )
                BC_Anime_genre_pearson_sim[i, j] = BC_corr

    return BC_Anime_genre_pearson_sim

BC_Anime_genre_pearson_sim = calculate_genre_pearson_sim(anime_data)

# Create a DataFrame to hold the correlation values
BC_Anime_genre_pearson_sim_df = pd.DataFrame(BC_Anime_genre_pearson_sim, index=anime_data['cleaned_Anime'], columns=anime_data['cleaned_Anime'])

# Create a Series to map anime titles to their indices
BC_Anime_genre_index_sim = pd.Series(anime_data.index, index=anime_data['cleaned_Anime']).drop_duplicates()

print("Shape of Anime_genre_pearson_sim: " + str(BC_Anime_genre_pearson_sim_df.shape))
print("==========================================================================")
print(BC_Anime_genre_pearson_sim_df)
print("==========================================================================")
print("Shape of Anime_genre_index_sim: " + str(BC_Anime_genre_index_sim.shape))
print("==========================================================================")
print(BC_Anime_genre_index_sim)

def BOW_Pearson_get_recommendations1(user_input):
    closest_match = find_closest_match(user_input, anime_data['cleaned_Anime'])
    idx = BC_Anime_desc_index_sim[closest_match]
    BC_sim_scores = list(enumerate(BC_Anime_desc_pearson_sim_df.iloc[idx]))
    BC_sim_scores = sorted(BC_sim_scores, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in BC_sim_scores]
    print("Closest match: " + closest_match)
    VerOne_Recommendations = anime_data.iloc[anime_indices]
    
    print(pd.DataFrame(VerOne_Recommendations[['Anime', 'cleaned_Genre', 'Description']]))
    
    return BOW_Pearson_get_recommendations2(VerOne_Recommendations, closest_match)

def BOW_Pearson_get_recommendations2(recommendations, closest_match):
    if len(recommendations) == 0:
        return "No recommendations available."
    
    anime_title = closest_match
    idx = 0
    
    BC_sim_scores = list(enumerate(BC_Anime_genre_pearson_sim_df.iloc[idx]))
    BC_sim_scores = sorted(BC_sim_scores, key=lambda x: x[1], reverse=True)
    anime_indices = [i[0] for i in BC_sim_scores]
    
    top_recommendations = anime_data.iloc[anime_indices]
    
    print('Anime title: ' + anime_title)
    
    return top_recommendations[0:11]

# Final version of anime recommendations using BOW and Pearson correlation for genre
user_input = input('Please enter one fo the favorite anime name for more recommendations: ')
RefinedVer_Recommendations = BOW_Pearson_get_recommendations1(user_input)
pd.DataFrame(RefinedVer_Recommendations[['Anime', 'cleaned_Genre', 'cleaned_Description']][0:4])
