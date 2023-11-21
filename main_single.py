
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue, Pool, TimeoutError
import queue as q
import numpy
import os
import random
HOST = "localhost"
WORKERS = 1
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

connections.connect("default",host=HOST)
BATCH_SIZE = 10
dim = 384
collection_name = "documents"
root = "data/posts"
data_files = os.listdir(root)
print(len(data_files))
# sen_count,line,file


BATCH = 1000




def process(row, rec_cnt):
    
   
    
    return rec_cnt
        
        

if __name__ == "__main__":

    DROP = True
    collection_name = "documents"
    connections.connect("default",host=HOST)


    has_collection = utility.has_collection(collection_name)
    print(f"Collection {collection_name} exists: {has_collection}")

    if has_collection and DROP:
        utility.drop_collection(collection_name)
        print(f"Collection dropped: {collection_name}")
        
        
    if DROP:    
        fields = [
            FieldSchema("pk", DataType.VARCHAR, is_primary=True,auto_id=False, max_length=100),
            FieldSchema("embeddings",DataType.FLOAT_VECTOR,dim=dim),
            FieldSchema("sentence", DataType.VARCHAR, max_length=65535),
            FieldSchema("doc_title", DataType.VARCHAR, max_length=65535),
            FieldSchema("sentence_id", DataType.INT64)
        ]

        schema = CollectionSchema(fields,"Document VectorDB")
        collection = Collection(collection_name, schema, consistency_level="Session")

    line_count = 0
    rec_cnt = 0
    ids = []
    lines = []
    files = []
    sen_counts = []
    vecs = []
    for f in data_files:
        with open(root+"/"+f, 'r') as fh:
            
            try:
                all_lines = fh.read().split(". ")
                line_count += len(all_lines)
                for sen_count, line in enumerate(all_lines):
                    rec_cnt += 1
                    if rec_cnt > BATCH:
                        
                        
                        
                        collection.insert([ids, vecs,lines,files, sen_counts])
                        collection.flush()
                        ids = []
                        lines = []
                        files = []
                        sen_counts = []
                        vecs = []
                        rec_cnt = 0
                        print(f"Row: {collection.num_entities}")
                
                    files.append(f)
                    sen_counts.append(sen_count)
                    ids.append(f+str(line_count))
                    lines.append(line)
                    vecs.append(numpy.reshape(model.encode(line),(dim)))
            except Exception as e:
                print("Error: ",e, f)
        print(f, "  done")
    
    if ids:
        row = [ids, vecs,lines,files, sen_counts]
        collection.insert(row)
        collection.flush()
    
    print("Lines to process ----> ",line_count)
         
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection = Collection(collection_name, consistency_level="Session")
    collection.create_index("embeddings",index)