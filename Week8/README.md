# Loan Dataset Chatbot with RAG Pipeline

This project implements a chatbot that answers questions about a loan dataset using a **Retrieval-Augmented Generation (RAG)** pipeline. The chatbot retrieves relevant context from the dataset and generates responses using a language model.

## Features
- **Document Embedding**: Uses the `mxbai-embed-large` model to generate embeddings for the dataset.
- **FAISS Index**: Stores document embeddings in a FAISS index for efficient similarity search.
- **Context Retrieval**: Retrieves the top-k most relevant documents based on the query.
- **Response Generation**: Generates answers using the `gemma:2b` language model.
- **Interactive Chatbot**: Allows users to ask questions interactively.

---

## Workflow

1. **Dataset Loading**:
   - The loan dataset is loaded from a CSV file (`Training Dataset.csv`).
   - Missing values are filled with empty strings.

2. **Document Preparation**:
   - Each row in the dataset is converted into a structured text format for embedding.

3. **Embedding Generation**:
   - The `mxbai-embed-large` model is used to generate embeddings for each document.

4. **FAISS Index**:
   - Document embeddings are stored in a FAISS index for fast similarity search.

5. **Query Pipeline**:
   - A query is embedded and searched against the FAISS index to retrieve the top-k most relevant documents.

6. **Response Generation**:
   - The `gemma:2b` model generates a response based on the query and retrieved context.

7. **Interactive Chatbot**:
   - Users can interact with the chatbot by asking questions, and it will provide answers based on the dataset.

---

## File Structure

- **code.ipynb**: Main notebook containing the implementation of the chatbot.
- **Dataset/Training Dataset.csv**: The loan dataset used for context retrieval.

---

## Requirements

Install the following Python libraries before running the code:

```bash
pip install pandas numpy faiss-cpu ollama
```

---

## How to Run

1. Place the dataset (`Training Dataset.csv`) in the `Dataset` folder.
2. Open the `code.ipynb` notebook in Jupyter or VS Code.
3. Run all the cells sequentially.
4. Interact with the chatbot by entering queries in the terminal.

---

## Example Usage

```plaintext
Ask a question: What is the loan amount for ID 12345?

🔍 Answer: The loan amount for ID 12345 is $10,000.
```

---

## Models Used

- **Embedding Model**: `mxbai-embed-large` for generating document embeddings.
- **Language Model**: `gemma:2b` for generating natural language responses.

---

## Future Enhancements

- Add support for larger datasets by chunking documents.
- Improve response quality by fine-tuning the language model.
- Add a web-based interface for easier interaction.

---

