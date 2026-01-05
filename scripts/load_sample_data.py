"""Simple loader: creates collection in Milvus, inserts sample docs, and builds simple KG."""
from backend.app.services.milvus_service import MilvusService
from backend.app.services.kg_service import KGService
from data.sample_data import SAMPLE_DOCS

def main():
    m = MilvusService()
    m.create_collection()
    print("Inserting sample documents into Milvus...")
    m.insert_documents(SAMPLE_DOCS)
    print("Building KG... (in-memory)")
    kg = KGService()
    print("Done. Sample data loaded.")

if __name__ == '__main__':
    main()
