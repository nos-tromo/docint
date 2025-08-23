from modules.rag import RAG

# Define the data directory and session parameters
data_dir = "data"
queries = [
    "Was sind die Gründe für die Geschlechterdivergenz?",
    # "In welchen Ländern wird das Phänomenfestgestellt?",
    # "Warum tendieren Männer stärker als Frauen dazu, autoritäre Positionen zu unterstützen?",
]
print(f"Data directory: {data_dir}")

# Initialize the RAG pipeline and process the documents
rag = RAG(qdrant_collection="testdb-cli")
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
