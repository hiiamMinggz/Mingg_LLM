import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from embedding import MyEmbeddingFunction
from ultis import normalizeString
class Ingestion():
    def __init__(self) -> None:
        pass

    def text_loader(self, data_path):
        """Text Loader for text file

        Args:
            data_path ([str, list[str]]): data_path can be file path or dir path or list of file path

        Raises:
            TypeError: Path must be path of text file or dir or list of file path
        """
        contents = []
        if isinstance(data_path, str):
            if os.path.isfile(data_path):
                with open(data_path, mode='r', encoding='utf-8') as f:
                    contents.append(f.read())
            elif os.path.isdir(data_path):
                for path in os.listdir(data_path):
                    if path.endswith('.txt'):
                        with open(os.path.join(data_path, path), mode='r', encoding='utf-8') as f:
                            contents.append(f.read())
        elif isinstance(data_path, list[str]):
            for path in data_path:
                if path.endswith('.txt'):
                    with open(path, mode='r', encoding='utf-8') as f:
                        contents.append(f.read())
        else:
            raise TypeError('Path must be path of text file or dir or list of file path')
        # format lai contents
        return normalizeString('\n'.join(contents))
    def text_spliter(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        recursive_spliter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=400,
            chunk_overlap=128,
            length_function=len,    
            # separators=[
            #         "\n\n",
            #         "\n",
            #         "?",
            #         ".",
            #         "...",
            #         ",",
            #         " ",
            #         "",
            #         ]
            )
        return recursive_spliter
    def vector_db(self, documents):
        """_summary_
        """
        emb_fn = MyEmbeddingFunction()
        client = chromadb.PersistentClient(path="chroma_db")
        client.clear_system_cache()
        collection = client.get_or_create_collection(name="medical_rag", embedding_function=emb_fn, metadata={"hnsw:space": "cosine"})
        print(collection.count())
        doc_ids = [str(id) for id,val in enumerate(documents)]
        print(len(documents))
        print(len(doc_ids))
        # print(doc_ids)
        collection.upsert(documents=documents, ids = doc_ids)
        print("Số lượng document", collection.count())
        # print("10 documents đầu tiên", collection.peek())
        # query = "bệnh lậu là gì"

        # # Include the source document and the Cosine Distance in the query result
        # result = collection.query(query_texts=[query],
        #                         n_results=5, include=["documents", 'distances',])

        # print("Query:", query)
        # print("Most similar sentences:")
        # # Extract the first (and only) list inside 'ids'
        # ids = result.get('ids')[0]
        # # Extract the first (and only) list inside 'documents'
        # documents = result.get('documents')[0]
        # # Extract the first (and only) list inside 'documents'
        # distances = result.get('distances')[0]

        # for id_, document, distance in zip(ids, documents, distances):
        #     # Cosine Similiarity is calculated as 1 - Cosine Distance
        #     print(f"ID: {id_}, Document: {document}, Similarity: {1 - distance}")

ingestion = Ingestion()
# for file in os.listdir('clean_corpus/'):
#     print(file)
content = ingestion.text_loader('benh_a_z')
# print(content)
chunks = ingestion.text_spliter().split_text(content)
# print(len(chunks))
# for chunk in chunks:
#     print(len(chunk))
# print(chunks[0:5])
ingestion.vector_db(chunks)