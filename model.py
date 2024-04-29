from prompts import classification_prompt, generation_prompt
import openai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import Pinecone
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
import http.client
import json

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv('PINECONE_API_KEY')


# Load the pre-trained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


# The name of the Pinecone index
index_name = "drug-index" 

# Create SentenceTransformer-based embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a Pinecone client using the existing index and SentenceTransformer embeddings
docsearch = Pinecone.from_existing_index(index_name, embeddings)

def get_similar_docs(query, k=1, score=False):
    """
    Retrieves similar documents based on the input query.

    Args:
        query (str): The input query string for which similar documents are to be retrieved.
        k (int, optional): The number of similar documents to be returned. Defaults to 1.
        score (bool, optional): Whether to include scores for the similarity. Defaults to False.

    Returns:
        list or dict: A list of similar documents' IDs (and scores if score=True) based on the query.
    """

    similar_docs = docsearch.max_marginal_relevance_search(query, k=k, fetch_k=10)

    return similar_docs

def content_filter(text):
    """
    Receives a text and extracts a list of labels from it using OpenAI.

    Args:
        text (str): The input text from which labels are to be extracted.

    Returns:
        list: A list of labels extracted from the input text.
    """
    in_progress = True
    counter = 0
    labels = []
    while in_progress:
        try:
            prompt = classification_prompt.format(text=text)

            answer = openai.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a classification model that classifies the text into labels: \
                              DRUG_INFO, HARMFUL_CONTENT, OTHER. \nKeeping in mind the prompt's label mapper and the examples",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=0.15,
            )

            labels = answer.choices[0].message.content

            labels = labels.split(", ")

            if len(labels) > 0:
                in_progress = False

        except Exception as prompt_exteption:
            print(
                "Exception while classification using OpenAi API", prompt_exteption
            )

        finally:
            if counter >= 10:
                in_progress = False
            counter = counter + 1
    return labels

def get_google_snippets(query):
    """
    Retrieves snippets from Google search results for the given query.

    Args:
        query (str): The search query for which snippets are to be retrieved.

    Returns:
        tuple: A tuple containing a list of snippets and a list of sources (links).
    """
    # Establish connection to the Google SERPer API
    conn = http.client.HTTPSConnection("google.serper.dev")

    # Prepare payload
    payload = json.dumps({
      "q": query
    })

    # Prepare headers
    headers = {
      'X-API-KEY': os.getenv('X-API-KEY'),
      'Content-Type': 'application/json'
    }

    # Send POST request
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()

    # Read response data
    data = res.read()

    # Parse JSON response
    data = json.loads(data)

    # Extract snippets from organic search results
    snippets = [result["snippet"] for result in data.get("organic", [])]

    sources = [result["link"] for result in data.get("organic", [])]

    return snippets,sources[:5]

def check_wikipedia_search(query):
    """
    Checks if there is a relevant Wikipedia search result for the given query.

    Args:
        query (str): The search query to be checked against Wikipedia.

    Returns:
        str or bool: The Wikipedia search result if found, or False if not found.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    result = wikipedia.run(query)
    if result == "No good Wikipedia Search Result was found":
        return False
    else:
        return result

def generate_response(
    inquiry_message,
    context,
):
    """
    Generates a response based on the given inquiry message and context.

    Args:
        inquiry_message (str): The inquiry message or prompt.
        context (list): A list of context items to provide additional information.

    Returns:
        str: The generated response based on the inquiry message and context.
    """
    prompt = generation_prompt.format(
        context=context,
        inquiry_message=inquiry_message,
    )

    in_progress = True
    counter = 0
    response = ""
    while in_progress:
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cool pharmaceutical representative known for delivering engaging, concise answers about drugs based on provided information.",
                    },
                    {"role": "user", "content": f"{prompt}"},
                ],
                temperature=0.20,
            )

            response = response.choices[0].message.content
            response = response.replace('"', "")
            if response.strip() != "":
                in_progress = False

        except Exception as response_exception:
            print(
                "Exception while generating response using openai",
                response_exception,
            )
        finally:
            if counter >= 10:
                in_progress = False
            counter = counter + 1

    return response



def final_function(text):
    """
    Performs a series of operations based on the input text and returns a final result.

    Args:
        text (str): The input text to be processed.

    Returns:
        str: The final result based on the processing of the input text.
    """
    classification = content_filter(text)
    context = []
    
    if "DRUG_INFO" in classification:
        vector_db_response = get_similar_docs(text)
        wikipedia_response = check_wikipedia_search(text)
        google_search_response, google_search_sources = get_google_snippets(text)
        
        if vector_db_response:
            context.append(vector_db_response)
        if wikipedia_response:
            context.append(wikipedia_response)
        if google_search_response:
            context.append(google_search_response)

        context = [item for sublist in context for item in sublist]
        result = generate_response(text, context)
        result = result.replace("\n\n", "")
        result = result.replace("\n", "")

        result = result.replace(": [", ": ")
        result = result.replace("[", "")
        result = result.replace("]. ", ". ")
        result = result.replace("].", ". ")
        result = result.replace("]", ". ")

        sources_list = "\n".join([f"{i+1}. {source}" for i, source in enumerate(google_search_sources)])
        sources = f"\n\nSources:\n{sources_list}"
        
        result += sources

    elif "HARMFUL_CONTENT" in classification:
        result = "I appreciate your inquiry, but I must respectfully decline to engage in discussions or activities that could potentially be harmful. If you have any other questions or topics related to drug or medicine you'd like to discuss, I'm more than happy to assist you with those."

    elif "OTHER" in classification:
        result = "I'm sorry, but that falls outside of my scope of expertise. However, if you have any questions related to drugs or medicines, I'm here to assist you."

    else:
        result = "I'm sorry, but that falls outside of my scope of expertise. However, if you have any questions related to drugs or medicines, I'm here to assist you."
    
    return result
