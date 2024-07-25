from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from docx import Document
import csv
import os

app = Flask(__name__)

# Function to read the document and extract text from a .docx file
def read_docx(file_path):
    """Read text from a .docx file and return it as a single string."""
    doc = Document(file_path)
    text = [para.text for para in doc.paragraphs]
    return "\n".join(text)

# Function to read the document and extract text from a .csv file
def read_csv(file_path):
    """Read text from a .csv file and return it as a single string."""
    text = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text.append(','.join(row))
    return "\n".join(text)

# Function to split the text into manageable chunks
def chunk_text(text, chunk_size=2000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Function to answer questions using the QA pipeline
def answer_question(question, context_chunks):
    if not question.strip():
        return "The question cannot be empty."

    answers = []
    for chunk in context_chunks:
        result = qa_pipeline(question=question, context=chunk)
        answers.append(result.get('answer', 'No answer found.'))

    combined_answer = " ".join(answers)
    return combined_answer if combined_answer else "No relevant information found."

# Initialize the question-answering pipeline with a pre-trained model
qa_pipeline = pipeline("question-answering")

# Determine the correct file path and type
file_path = 'C:\\chatbot\\documents\\DocumenTATION ADMIN.csv.docx'  # Adjust this path as needed

# Check if the file is a .docx or .csv
if file_path.endswith('.docx'):
    document_text = read_docx(file_path)
elif file_path.endswith('.csv'):
    document_text = read_csv(file_path)
else:
    raise ValueError("Unsupported file type")

# Split the document text into chunks
chunks = chunk_text(document_text)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    if question:
        answer = answer_question(question, chunks)
        return jsonify({'answer': answer})
    return jsonify({'answer': 'Please ask a question.'})

if __name__ == "__main__":
    app.run(debug=True)
