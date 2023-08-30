# Description

This extension provides a simple user interface for generating text by providing contextual information based on the user's input. 


## UI

The UI consists of several components:

- **Settings**: The settings for the ChromaDB client ad retrieval. You can specify the host, port, and collection name of the ChromaDB server. You can also enable or disable the extension, specify the number of documents to retrieve (k), and the relevance threshold for the documents retrieved.

- **Connect to Chroma DB**: This button connects to the ChromaDB server with the specified settings. The status of the connection is displayed above the button.

## Note

This extension uses a language model for text generation and ChromaDB for providing contextual documents. Make sure you have a running ChromaDB server.