
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Pool
import numpy
import random
import openai
import logging
import datetime
import csv
import time
import os

logging.basicConfig(filename="session_log.txt",filemode="a", level=logging.INFO)
TIMESTAMP = datetime.datetime.now()
logging.info(f"\n\n=============================\nSession: {TIMESTAMP}")
collection_name = "documents"
connections.connect("default",host="localhost")
openai.organization = os.getenv("OAI_ORG")
openai.api_key = os.getenv("OAI_KEY")


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


USE_GPT4 = True


def fetch_openai(query, docs):
   
    doc = "\n".join(docs)
    GPT3 = "gpt-3.5-turbo"  
    GPT4 = "gpt-4-0613"
    GPT_VERSION = GPT3
    if USE_GPT4:
        GPT_VERSION = GPT4
  
    messages = [
          {
                "role":"user", "content": query
            }
    ]

    if docs:
        logging.info(f"Using documents (RAG): {len(docs)}")
        messages.append({
                "role":"user", "content": f"documents to be used: {doc}"
            })
        messages.append({
                "role":"system", "content": "ONLY use the documents in the user query to answer"
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
    run_data = [usage.completion_tokens,  usage.prompt_tokens, usage.total_tokens, response.model, len(docs), query]
    return extract_message_content(response), run_data

def extract_message_content(response):
    logging.info(response)
    return response.choices[0].message.content


def build_index():
    index = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
    has_collection = utility.has_collection(collection_name)
    print(f"Collection {collection_name} exists: {has_collection}")

    if has_collection:
        collection = Collection(collection_name, consistency_level="Session")
        collection.create_index("embeddings",index)
    print(f"Collection exists: {collection_name}, {collection.num_entities}")

def record_run(data):
    with open("run.csv","a") as fh:
        csv.writer(fh).writerow(data)

def query_vecdb(query, limit=3):
    
    vectors_to_search = numpy.reshape(model.encode(query),(384))

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 20},
    }
    print("Loading collection")
    collection = Collection(collection_name, consistency_level="Session")
    collection.load()
    print("Done.. searching")
    result = collection.search([vectors_to_search],anns_field="embeddings", param=search_params, limit=limit, output_fields=["sentence","pk","doc_title"])
    file_list = []
    for hits in result:
        
        for h in hits:
            file_list.append(h.entity.get("doc_title"))

    DOCS = []
    REFS = set(file_list)

    for f in REFS:
        DOCS.append(open("data/posts/"+f,"r").read())
    
    print(f"Hits: {len(REFS)}")
    return REFS, DOCS, hits.distances

if __name__ == "__main__":

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

    for q in querys:
        docs = []
        refs = []
        
        refs, docs, distances = query_vecdb(q, limit=5)
        print("Query: ", q)
        print("Response -------------------")
    
        for d in docs:
            print(">",d[0:200])
        print("\nLLM Response:")
        #docs = []
        rag_res, rag_run = fetch_openai(q, docs)
        print("#RAG: ",rag_res)
        res, run = fetch_openai(q,[])
        print()
        print("#Non-RAG: ",res)

        v_rag = model.encode(rag_res)
        v = model.encode(res)
        similarity_score = util.dot_score(v_rag,v)[0][0].item()
        logging.info(f"Dot Score: {similarity_score}\n\n==========================")
        
        rag_run.extend(["RAG",similarity_score,numpy.mean(distances),numpy.std(distances)])
        run.extend(["PROMPT_ONLY",similarity_score,0,0])
        record_run(rag_run)
        record_run(run)
        time.sleep(30)
        

        

