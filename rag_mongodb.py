from datasets import load_dataset
from dotenv import dotenv_values
import pymongo
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

dataset_df_embedded_values = dataset_df.copy()
# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with the new OpenAI embedding Model "text-embedding-3-small"
dataset_df = dataset_df.drop(columns=['plot_embedding'])
print(dataset_df.head()['plot'])

#Importing OpenAI apikey
loaded_secrets = dotenv_values(".env")
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
        # embedding = openai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        # return embedding
        print("Simulating the request to OPENAI")
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return None
    
#Vector Search Function
def vector_search(user_query, collection):
    """
    Performs a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."
    
    #Define the vector search pipeline for the aggregation operation
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "plot_embedding_optimised",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 5  # Return top 5 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "plot_embedding_opitimzed": 0,  # Exclude the plot_embedding_opitimzed field
                "plot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1, # Include the genres field
                "score": {
                    "$meta": "vectorSearchScore"  # Include the search score
                }
            }
        }
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)

#Set a valid MongoDB connection
def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        #Create a new client and connect to the server
        mdb_client = pymongo.MongoClient(mongo_uri)
        print("Pinged your deployment. You successfully connected to MongoDB")
        return mdb_client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
    

#dataset_df["plot_embedding_optimised"] = dataset_df['plot'].apply(get_embedding)
dataset_df["plot_embedding_optimised"] = dataset_df_embedded_values['plot_embedding']

mongo_uri = loaded_secrets["ATLAS_URI"]
if not mongo_uri:
    print("ATLAS_URI not set in environment variables")

mdb_client = get_mongo_client(mongo_uri)

#Ingest data into MongoDB
db = mdb_client['movies_embedded']
collection = db['movie_collection']

documents = dataset_df.to_dict('records')
collection.insert_many(documents)

print("Data ingestion into MongoDB completed")

