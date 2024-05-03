from flask import Flask, render_template, request, jsonify, session
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from docx import Document
from doc2docx import convert
from werkzeug.utils import secure_filename
from opencc import OpenCC
import os



os.environ['OPENAI_API_KEY']='OPENAI_API_KEY'
app = Flask(__name__)


chat_history = []
data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input')
    if not user_input:
        return jsonify({'error': 'No user input provided'})
    if user_input:
        embeddings = OpenAIEmbeddings()
        dir_path="../db/"
        new_db=None
        for db in os.listdir(dir_path):
            print(f"db name:{db}")
            db=FAISS.load_local(os.path.join(dir_path,db),embeddings, allow_dangerous_deserialization=True)
            if new_db is None:
                new_db = db
            else:
                new_db.merge_from(db)
        
        docs = new_db.similarity_search(user_input)
        llm = ChatOpenAI(
            model_name="gpt-4-turbo-2024-04-09",
            temperature=0.2
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.invoke({"input_documents": docs,"question":user_input}, return_only_outputs=True)
        cc = OpenCC('s2t')
        answer=cc.convert(response['output_text'])
        
        chat_history.append({'user': user_input, 'assistant': response['output_text']})
        return jsonify({'response': answer})

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host='0.0.0.0')  

