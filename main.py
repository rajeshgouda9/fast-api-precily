# importing libraries for api deployment and sentence_transformers for getting the tensor values
from fastapi import FastAPI
import uvicorn
from sentence_transformers import SentenceTransformer, util

# creating a model instance for a pretrained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.max_seq_length = 512
app = FastAPI()


@app.get('/')
def home():
    return {'text': 'sentence similarity'}


@app.get('/similarity')
def similarity(text1: str, text2: str):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    a = cosine_scores.item()
    return {'similarity score': round(a, 2)}


if __name__ == '__main__':
    uvicorn.run(app)
