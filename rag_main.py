
from pymilvus import (
    connections,
    utility,
    Collection,
)
from sentence_transformers import SentenceTransformer, util
import numpy
import openai
import logging
import datetime
import csv
import time
import os

#Setup logging
logging.basicConfig(filename="session_log.txt",
                    filemode="a", level=logging.INFO)
TIMESTAMP = datetime.datetime.now()
logging.info(f"\n\n=============================\nSession: {TIMESTAMP}")

#Milvus collection and connection
collection_name = "documents"
connections.connect("default", host="localhost")

#OpenAI Key
openai.organization = os.getenv("OAI_ORG")
openai.api_key = os.getenv("OAI_KEY")

#Initialise sentence transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


USE_GPT4 = True

## Fetch from Open AI - query and RAG docs
def fetch_openai(query, docs):

    doc = "\n".join(docs)
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4-0613"
    GPT_VERSION = GPT3
    if USE_GPT4:
        GPT_VERSION = GPT4

    messages = [
        {
            "role": "user", "content": query
        }
    ]

    #Only include documents in prompt if we are using RAG
    if docs:
        logging.info(f"Using documents (RAG): {len(docs)}")
        messages.append({
            "role": "user", "content": f"documents to be used: {doc}"
        })
        messages.append({
            "role": "system", "content": "ONLY use the documents in the user query to answer"
        })

    
    response = openai.ChatCompletion.create(
        model=GPT_VERSION,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[]
    )

    usage = response.usage

    #To compile run data
    run_data = [usage.completion_tokens,  usage.prompt_tokens,
                usage.total_tokens, response.model, len(docs), query]
    
    #return response text and run data
    return extract_message_content(response), run_data

## Extract response from Open AI API Response object
def extract_message_content(response):
    logging.info(response)
    return response.choices[0].message.content


## Record the run in 'run.csv' file
def record_run(data):
    with open("run.csv", "a") as fh:
        csv.writer(fh).writerow(data)

## Query Milvus Vec DB to retrieve similar documents
def query_vecdb(query, limit=3):

    #reshaping to search, pymilvus can't deal with missing dimensions
    vectors_to_search = numpy.reshape(model.encode(query), (384))

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 20},
    }
    print("Loading collection")
    collection = Collection(collection_name, consistency_level="Session")
    collection.load()
    print("Done.. searching")
    result = collection.search([vectors_to_search], anns_field="embeddings",
                               param=search_params, limit=limit, output_fields=["sentence", "pk", "doc_title"])
    file_list = []
    for hits in result:

        for h in hits:
            file_list.append(h.entity.get("doc_title"))

    DOCS = [] #Text from the blog posts
    REFS = set(file_list) #list of relevant blog posts (1 blog post = 1 txt file)

    for f in REFS:
        DOCS.append(open("data/posts/"+f, "r").read())

    print(f"Hits: {len(REFS)}")

    #return file references, text from blog posts and distance scores
    return REFS, DOCS, hits.distances


if __name__ == "__main__":
    # Queries for RAG/Non-RAG testing using blog posts
    # To get a good spread of results we should have a combination of queries that are relevant to the blog posts and those that are not.
    querys = ["Tell me about break dance",
              "How does Java compare with scala and clojure",
              "Tell me about south delhi, india",
              "What is the book written by JRR Tolkien",
              "What are the gifts given to india by the US",
              "What can you tell me about Restriced Boltzmann machines?",
              "Tell me about global variables in Tibco business works",
              "Tell me about percieved value price",
              "Tell me about the land registry price paid data set",
              "Tell me about Prototypes in Javascript",
              "Tell me about 5G-PPP",
              "Tell me about the Tibco Action Processor",
              "Tell me about the Zodiac FX OpenFlow switch"]

    # We run each query in the list above and get RAG and non-RAG results which we then compare
    for q in querys:
        docs = []
        refs = []

        # Retrieve documents similar to the query from the vector db
        # returns - references, documents, and distance scores
        refs, docs, distances = query_vecdb(q, limit=5)

        print("Query: ", q)
        print("Response -------------------")

        # Extract of blog posts retrieved using RAG
        for d in docs:
            print(">", d[0:200])
        print("\nLLM Response:")

        # Get response from GPT-4 with documents (RAG)
        rag_results, rag_run = fetch_openai(q, docs)
        print("#RAG: ", rag_results)

        # Get response from GPT-4 without documents (Non-RAG)
        non_rag_results, run = fetch_openai(q, [])
        print()
        print("#Non-RAG: ", non_rag_results)

        # Convert results to vectors
        v_rag = model.encode(rag_results)
        v = model.encode(non_rag_results)

        # Get similarity score
        similarity_score = util.dot_score(v_rag, v)[0][0].item()

        logging.info(
            f"Dot Score: {similarity_score}\n\n==========================")

        # Collect data from the RAG and Non-RAG runs and record results

        rag_run.extend(["RAG", similarity_score, numpy.mean(
            distances), numpy.std(distances)])
        run.extend(["PROMPT_ONLY", similarity_score, 0, 0])
        record_run(rag_run)
        record_run(run)

        # Wait a bit before the next call to OpenAI - can be removed
        time.sleep(30)
