from flask import Flask, render_template, request, jsonify
from docx import Document
import os
import google.generativeai as genai
import pandas as pd
import clickhouse_connect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()
app = Flask(__name__)  # Initialize Flask app


# Function to load and split a DOCX file
def load_docx(_file_path):
    class DocxLoader:
        def __init__(self, file_path_):
            self.file_path = file_path_
            self.document = Document(file_path_)  # Open DOCX file

        def load_and_split(self, split_by_paragraphs=5):
            # Extract non-empty paragraphs from the document
            paragraphs = [paragraph.text for paragraph in self.document.paragraphs if paragraph.text.strip()]
            # Split paragraphs into chunks
            chunks = [paragraphs[g:g + split_by_paragraphs] for g in range(0, len(paragraphs), split_by_paragraphs)]
            # Join chunks into text blocks
            split_texts = ["\n".join(chunk) for chunk in chunks]
            return split_texts

    loader = DocxLoader(_file_path)  # Create DocxLoader instance
    pages = loader.load_and_split(split_by_paragraphs=5)  # Load and split document
    text = "\n".join(pages)  # Combine text chunks into a single string
    # Initialize text splitter to handle long texts
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs_ = text_splitter.create_documents([text])  # Split text into manageable chunks
    for e, d in enumerate(docs_):
        d.metadata = {"doc_id": e}  # Assign metadata to each document chunk
    return docs_


# Retrieve and configure the Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)  # Configure Google Generative AI with the API key


# Function to get embeddings from text using the Gemini API
def get_embeddings(text):
    model = 'models/embedding-001'
    embedding = genai.embed_content(model=model, content=text, task_type="retrieval_document")
    return embedding['embedding']


# Path to the DOCX file
file_path = 'C:\\chatbot\\documents\\DocumenTATION ADMIN.docx'
docs = load_docx(file_path)  # Load and process DOCX file
content_list = [doc.page_content for doc in docs]  # Extract content from documents
# Generate embeddings for each content chunk
embeddings = [get_embeddings(content) for content in content_list]

# Create a dataframe to prepare data for database ingestion
dataframe = pd.DataFrame({
    'page_content': content_list,
    'embeddings': embeddings
})
# Filter embeddings to ensure they have the correct length
dataframe['embeddings'] = dataframe['embeddings'].apply(lambda x: x if len(x) == 768 else None)
dataframe = dataframe.dropna(subset=['embeddings'])  # Remove rows with invalid embeddings

# Initialize ClickHouse client
client = clickhouse_connect.get_client(
    host='msc-92962d0f.us-east-1.aws.myscale.com',
    port=443,
    username='zikirajae_org_default',
    password='passwd_rIaUftQz0QLBQt'
)

# Uncomment to create the table in ClickHouse
# client.command("""
# CREATE TABLE default.chatbot (
#  id Int64,
# page_contentString,
# embeddings Array(Float32),
# CONSTRAINT check_data_length CHECK length(embeddings) = 7768
# ) ENGINE = MergeTree()
# ORDER BY id
# """)

# Insert data into ClickHouse in batches
# batch_size = 10
# num_batches = len(dataframe) // batch_size
# for i in range(num_batches):
#    start_idx = i * batch_size
#   end_idx = start_idx + batch_size
#    batch_data = dataframe[start_idx:end_idx]
# Insert batch data into the table
#   client.insert("default.chatbot", batch_data.to_records(index=False).tolist(),
#                column_names=batch_data.columns.tolist())
#   print(f"Batch {i + 1}/{num_batches} inserted.")

# Uncomment to create a vector index for fast retrieval (make sure to use a unique index name)
# client.command("""
# ALTER TABLE default.chatbot
#   ADD VECTOR INDEX vector_index embeddings
# TYPE MSTG
# """)


# Function to retrieve relevant documents based on a user query
def get_relevant_docs(user_query):
    query_embeddings = get_embeddings(user_query)
    results = client.query(f"""
        SELECT page_content,
        distance(embeddings, {query_embeddings}) as dist FROM default.chatbot ORDER BY dist LIMIT 3
    """)
    relevant_docs = [row['page_content'] for row in results.named_results()]
    return relevant_docs


# Function to construct a prompt for the Generative AI model
def make_rag_prompt(query, relevant_passage):
    relevant_passage = ' '.join(relevant_passage)  # Combine relevant passages into a single string
    prompt = (
        f"You are a helpful and informative chatbot that answers questions using text from the reference passage included below. "
        f"Respond in a complete sentence and make sure that your response is easy to understand for everyone. "
        f"Maintain a friendly and conversational tone. If the passage is irrelevant, feel free to ignore it.\n\n"
        f"QUESTION: 'user:{query}'\n"
        f"PASSAGE: '{relevant_passage}'\n\n"
        f"ANSWER:"
    )
    return prompt


# Function to generate a response from the Generative AI model
def generate_response(user_prompt):
    model = genai.GenerativeModel('gemini-pro')
    answer = model.generate_content(user_prompt)
    return answer.text


# Function to generate an answer for a user query
def generate_answer(query):
    relevant_text = get_relevant_docs(query)
    text = " ".join(relevant_text)
    prompt = make_rag_prompt(query, relevant_passage=relevant_text)
    answer = generate_response(prompt)
    return answer


# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')


# Route for handling user queries
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['query']
    answer = generate_answer(user_query)
    return jsonify({'answer': answer})


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
