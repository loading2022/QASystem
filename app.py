from flask import Flask, render_template, request, jsonify
import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from PyPDF2 import PdfReader
from docx import Document
from doc2docx import convert
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import shutil
import pymysql
from difflib import SequenceMatcher


connect_db = pymysql.connect(
    host='localhost',
    port=your_port,
    user='your_username',
    passwd='your_password',
    charset='utf8',
    db='your_dbname'
)

with connect_db.cursor() as cursor:
    sql = """
    CREATE TABLE IF NOT EXISTS userQ(
        ID int NOT NULL AUTO_INCREMENT PRIMARY KEY,
        User_Q varchar(100)
    );
    """
    
    # 執行 SQL 指令
    cursor.execute(sql)
    
    # 提交至 SQL
    connect_db.commit()


os.environ['OPENAI_API_KEY']='your_api_key'
app = Flask(__name__)

CORS(app)
chat_history = []
top_three=[]
data_folder = os.path.join(os.path.dirname(__file__), 'data')
shutil.rmtree(data_folder)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


@app.route('/')
def index():
#return 'HI'
    return render_template('index.html', chat_history=chat_history,top_three=top_three)

extension_list=[]
filename_list=[]
require_extension=['.docx','.pdf','.doc']
@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'filename' in request.files:
        files = request.files.getlist('filename')
        for file in files:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(os.path.dirname(__file__), 'data', filename)
                _, extension = os.path.splitext(file_path)
                print(extension)
                if filename not in filename_list and (extension in require_extension):
                    print('ok')
                    file.save(file_path)
                    extension_list.append(extension)
                    filename_list.append(filename)
        print(filename_list)
        print("File path:", file_path)
        return jsonify({'filenames': filename_list, 'extensions': extension_list})
    return 'No file uploaded.'

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form.get('user_input')
    
    if not user_input:
        return jsonify({'error': 'No user input provided'})
    if user_input:
        similarities=[]
        with connect_db.cursor() as cursor:
            sql = """SELECT user_q from userq"""
            cursor.execute(sql)
            data = cursor.fetchall()
            for d in data:
                text = d[0]
                ratio = similarity(text, user_input)
                if ratio>0 and ratio<1:
                    print(text,ratio)
                    similarities.append(text)

            sql = "INSERT IGNORE INTO userQ (User_Q) VALUES (%s);"
            cursor.execute(sql, (user_input,))
            connect_db.commit()
        
        temp = sorted(similarities, key=lambda x: x[1])[:3]
        top_three=[]
        for item in temp:
            top_three.append({'Q':item})
        print(top_three)
        files_folder = 'data'
        text = ""
        for filename in os.listdir(files_folder):
            file_path = os.path.join(files_folder, filename)
            if filename.endswith('.pdf'):
                text += get_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                text += get_text_from_docx(file_path)
            elif filename.endswith('.doc'):
                output_path=os.path.splitext(file_path)[0] + "_output.docx"
                if not os.path.exists(output_path):
                    convert(file_path,output_path)
                    os.remove(file_path)
                text += get_text_from_docx(output_path)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        docs = knowledge_base.similarity_search(user_input)

        llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.4
        )

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_input)

        response = {'response': response}

        chat_history.append({'user': user_input, 'assistant': response['response']})
        return jsonify({'response': response, 'top_three': top_three})

if __name__ == '__main__':
    app.run(debug=True)
    connect_db.close()    

