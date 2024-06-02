import chromadb
from embedding import MyEmbeddingFunction
class Retrieval():
    def __init__(self):
        pass
    def db_query(self,user_query, k=3, window_size=3):
        client = chromadb.PersistentClient(path="chroma_db")
        emb_fn = MyEmbeddingFunction()
        collection = client.get_collection(name="medical_rag", embedding_function=emb_fn)
        # print(f'DB chứa {collection.count()} văn bản')
        result = collection.query(query_texts=[user_query],
                                n_results=k, include=["documents", 'distances',])

        print("Query:", user_query)
        # print("Most similar sentences:")
        # Extract the first (and only) list inside 'ids'
        ids = result.get('ids')[0]
        # print(ids)
        window_id = []
        for _id in ids:
            window_id.append(str(int(_id) - int((window_size-1)/2)))
            window_id.append(str(int(_id) + int((window_size-1)/2)))
        window_content = ''.join(collection.get(sorted(window_id+ids))['documents'])
        return (window_content)
        # print(collection.get(['19634']))
        # Extract the first (and only) list inside 'documents'
        # documents = result.get('documents')[0]
        # Extract the first (and only) list inside 'documents'
        # distances = result.get('distances')[0]

        # for id_, document, distance in zip(ids, documents, distances):
            # Cosine Similiarity is calculated as 1 - Cosine Distance
            # print(f"ID: {id_}, Document: {document}, Similarity: {1 - distance}")

# my_retrieval = Retrieval()
# my_retrieval.db_query(user_query='cách điều trị bệnh trĩ')
        