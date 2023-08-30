from flask import Flask, render_template, request
import urllib.request
import fitz
import re
import numpy as np
import tensorflow_hub as hub
import openai
import os
from sklearn.neighbors import NearestNeighbors
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Rest of your existing code (download_pdf, preprocess, pdf_to_text, text_to_chunks, SemanticSearch class, etc.)
def download_pdf(url, output_path):
    urllib.request.urlretrieve(url, output_path)


def preprocess(text):
    text = text.replace('\n', ' ')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages

    text_list = []

    for i in range(start_page-1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return text_list


def text_to_chunks(texts, word_length=150, start_page=1):
    text_toks = [t.split(' ') for t in texts]
    page_nums = []
    chunks = []
    
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), word_length):
            chunk = words[i:i+word_length]
            if (i+word_length) > len(words) and (len(chunk) < word_length) and (
                len(text_toks) != (idx+1)):
                text_toks[idx+1] = chunk + text_toks[idx+1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx+start_page}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks


class SemanticSearch:
    
    def __init__(self):
        self.use = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
        self.fitted = False
    
    
    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_text_embedding(data, batch=batch)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True
    
    
    def __call__(self, text, return_data=True):
        inp_emb = self.use([text])
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]
        
        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors
    
    
    def get_text_embedding(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i:(i+batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings



def load_recommender(path, start_page=1):
    global recommender
    texts = pdf_to_text(path, start_page=start_page)
    chunks = text_to_chunks(texts, start_page=start_page)
    recommender.fit(chunks)
    return 'Corpus Loaded.'

def generate_text(openAI_key, prompt, model="gpt-3.5-turbo"):
    openai.api_key = openAI_key
    temperature=0.7
    max_tokens=256
    top_p=1
    frequency_penalty=0
    presence_penalty=0

    if model == "text-davinci-003":
        completions = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens,
            n=1,
            stop=None,
            temperature=temperature,
        )
        message = completions.choices[0].text
    else:
        message = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "assistant", "content": "Here is some initial assistant message."},
                {"role": "user", "content": prompt}
            ],
            temperature=.3,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        ).choices[0].message['content']
    return message

  
def generate_answer(question, openAI_key, model):
    topn_chunks = recommender(question)
    prompt = 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using [ Page Number] notation. "\
              "Only answer what is asked. The answer should be short and concise. \n\nQuery: "
    
    prompt += f"{question}\nAnswer:"
    answer = generate_text(openAI_key, prompt, model)
    return answer


def question_answer(chat_history, url, file, question, openAI_key, model):
    try:
        if openAI_key.strip() == '':
            return '[ERROR]: Please enter your Open AI Key.'
        if url.strip() == '' and file.filename == '':
            return '[ERROR]: Provide at least one of URL or PDF.'
        if url.strip() != '' and file.filename != '':
            return '[ERROR]: Provide only one of URL or PDF.'
        if model is None or model == '':
            return '[ERROR]: Select an LLM model.'
        
        if url.strip() != '':
            glob_url = url
            download_pdf(glob_url, 'corpus.pdf')
            load_recommender('corpus.pdf')
        else:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                load_recommender(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return '[ERROR]: Invalid file type'
        
        if question.strip() == '':
            return '[ERROR]: Question field is empty'
        
        if model == "text-davinci-003" or model == "gpt-4" or model == "gpt-4-32k":
            answer = generate_answer_text_davinci_003(question, openAI_key)
        else:
            answer = generate_answer(question, openAI_key, model)
        
        chat_history.append([question, answer])
        return chat_history
    except openai.error.InvalidRequestError as e:
        return f'[ERROR]: Either you do not have access to GPT4 or you have exhausted your quota!'



def generate_text_text_davinci_003(openAI_key,prompt, engine="text-davinci-003"):
    openai.api_key = openAI_key
    completions = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.7,
    )
    message = completions.choices[0].text
    return message


def generate_answer_text_davinci_003(question,openAI_key):
    topn_chunks = recommender(question)
    prompt = ""
    prompt += 'search results:\n\n'
    for c in topn_chunks:
        prompt += c + '\n\n'
        
    prompt += "Instructions: Compose a comprehensive reply to the query using the search results given. "\
              "Cite each reference using [ Page Number] notation (every result has this number at the beginning). "\
              "Citation should be done at the end of each sentence. If the search results mention multiple subjects "\
              "with the same name, create separate answers for each. Only include information found in the results and "\
              "don't add any additional information. Make sure the answer is correct and don't output false content. "\
              "If the text does not relate to the query, simply state 'Found Nothing'. Ignore outlier "\
              "search results which has nothing to do with the question. Only answer what is asked. The "\
              "answer should be short and concise. \n\nQuery: {question}\nAnswer: "
    
    prompt += f"Query: {question}\nAnswer:"
    answer = generate_text_text_davinci_003(openAI_key, prompt,"text-davinci-003")
    return answer

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# pre-defined questions
questions = [
    "What did the study investigate?",
    "Can you provide a summary of this paper?",
    "what are the methodologies used in this study?",
    "what are the data intervals used in this study? Give me the start dates and end dates?",
    "what are the main limitations of this study?",
    "what are the main shortcomings of this study?",
    "what are the main findings of the study?",
    "what are the main results of the study?",
    "what are the main contributions of this study?",
    "what is the conclusion of this paper?",
    "what are the input features used in this study?",
    "what is the dependent variable in this study?",
]


# Set the upload folder for storing uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

recommender = SemanticSearch()

ALLOWED_EXTENSIONS = {'pdf'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        openAI_key = request.form['openAI_key']
        url = request.form['url']
        file = request.files['file']
        question = request.form['question']
        model = request.form['model']
        
        chat_history = []
        result = question_answer(chat_history, url, file, question, openAI_key, model)
        
        if isinstance(result, str):
            return render_template('index.html', chat_history=chat_history, error=result)
        
        return render_template('index.html', chat_history=chat_history)
    
    return render_template('index.html', chat_history=[])

if __name__ == '__main__':
    app.run(debug=True)
