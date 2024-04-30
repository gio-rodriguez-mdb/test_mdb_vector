from datasets import load_dataset
from dotenv import load_dotenv
import os
import pandas as pd
import openai

# <https://huggingface.co/datasets/AIatMongoDB/embedded_movies>
dataset = load_dataset("AIatMongoDB/embedded_movies")

#Loading the dataset to a pandas dataframe
dataset_df = pd.DataFrame(dataset['train'])

#preview of the dataframe 
print(dataset_df.head())

# Remove data point where plot column is missing
dataset_df = dataset_df.dropna(subset=['plot'])
print("Number of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with the new OpenAI embedding Model "text-embedding-3-small"
dataset_df = dataset_df.drop(columns=['plot_embedding'])
print(dataset_df.head()['plot'])

#Importing OpenAI apikey
loaded_secrets = load_dotenv(".env")
openai.api_key = loaded_secrets["API_KEY"]

#Selecting the OpenAI embedding model
EMBEDDING_MODEL = "text-embedding-3-small"

#Get embedding values
def get_embedding(text):
    """Generate an embedding for the given text using OpenAI's API."""

    # Check for valid input
    if not text or not isinstance(text, str):
        return None

    try:
        # Call OpenAI API to get the embedding
        embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
    
dataset_df["plot_embedding_optimised"] = dataset_df['plot'].apply(get_embedding)
print(dataset_df.head()['plot_embedding_optimised'])    