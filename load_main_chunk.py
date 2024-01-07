from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from sentence_transformers import SentenceTransformer
import numpy
import os


HOST = "localhost"

#because sentence transformers outputs vector of dim (384,)
dim = 384

#setup vectorising model - sentence transformers
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#connect to milvus
connections.connect("default", host=HOST)

#colection name
collection_name = "documents_lines"

#root file location for blog posts - change this to point to your own files
root = "data/posts"

#data files to load data from
data_files = os.listdir(root)
print(len(data_files))

#batch size
BATCH = 1000

if __name__ == "__main__":

    #To drop and recreate or not...
    DROP = True

    has_collection = utility.has_collection(collection_name)
    print(f"Collection {collection_name} exists: {has_collection}")

    if has_collection and DROP:
        utility.drop_collection(collection_name)
        print(f"Collection dropped: {collection_name}")

    if DROP:
        #Recreate vector database if we dropped it.. start with schema
        fields = [
            FieldSchema("pk", DataType.VARCHAR, is_primary=True,
                        auto_id=False, max_length=100),
            FieldSchema("embeddings", DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema("sentence", DataType.VARCHAR, max_length=65535),
            FieldSchema("doc_title", DataType.VARCHAR, max_length=65535),
            FieldSchema("doc_title", DataType.INT64),
            FieldSchema("sentence_id", DataType.INT64)
        ]

        schema = CollectionSchema(fields, "Document VectorDB")
        collection = Collection(collection_name, schema,
                                consistency_level="Session")

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

                        collection.insert(
                            [ids, vecs, lines, files, sen_counts])
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
                    vecs.append(numpy.reshape(model.encode(line), (dim)))
            except Exception as e:
                print("Error: ", e, f)
        print(f, "  done")

    if ids:
        row = [ids, vecs, lines, files, sen_counts]
        collection.insert(row)
        collection.flush()

    print("Lines to process ----> ", line_count)

    #create index on 'embeddings' column
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection = Collection(collection_name, consistency_level="Session")
    collection.create_index("embeddings", index)
