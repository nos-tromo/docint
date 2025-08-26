from pathlib import Path

from wizard.modules.rag import RAG
from wizard.utils.logging_cfg import setup_logging

# Define paths and initialize RAG
data_dir = "data"
q_dir = Path("helpers") / "queries.txt"
rag = RAG()

choice = input("Select existing collection? (y/n): ").strip().lower()
col_name = input("Enter collection name: ").strip()
if choice == 'y':
    col_path = Path.home() / ".qdrant" / "storage" / "collections" / col_name
    if col_name:
        if not col_path.exists():
            print(f"Collection path {col_path} does not exist. Exiting.")
            exit(1)
        print(f"Using existing collection: {col_name}")
    else:
        print("No collection name provided. Exiting.")
        exit(1)
elif choice == 'n':
    print(f"Creating new collection: {col_name}.")
else:
    print("Invalid choice. Please enter 'y' or 'n'. Exiting.")
    exit(1)


with open(q_dir, "r", encoding="utf-8") as file:
    queries = [line.strip() for line in file if line.strip()]

print(f"Data directory: {data_dir}")

# Initialize the RAG pipeline and process the documents
rag.ingest_docs(data_dir)
print("Documents ingested successfully.")

# Initialize the session and start it
rag.init_session_store()
rag.start_session()
print("Session started successfully.")

# Process the queries and print the responses
for index, query in enumerate(queries, start=1):
    result = rag.chat(query)
    print("----------------------")
    query = result.get("query", "No query found in response.")
    response = result.get("response", "No response found in response.")
    sources = [
        source.get("text", "No source found") for source in result.get("sources", [])
    ]
    print(f"Query {index}: {query}")
    print("----------------------")
    print(f"Response {index}: {response}")
    print("----------------------")
    print("Sources:")
    for index, text in enumerate(sources, start=1):
        print(index, text)
    print("----------------------")
print("All queries processed successfully.")

# Export the session data to a JSON file
rag.export_session()
print("Session data exported successfully.")
