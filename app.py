
import cv2
from fastapi import FastAPI, File, UploadFile
from starlette.config import Config
from typing import List
import os
import pickle
import numpy as np
import uvicorn
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add


app = FastAPI()

with open('captions_convert_token.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('model.h5')

vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

max_length = 35
vocab_size = len(tokenizer.word_index) + 1

def generate_caption(image, model, tokenizer, max_length):
    try:
        image = preprocess_input(image)
        image_features = vgg_model.predict(image, verbose=0)
        in_text = 'startseq'
        for _ in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = model.predict([image_features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = tokenizer.index_word[yhat]
            in_text += ' ' + word
            if word == 'endseq':
                return in_text.replace('startseq', '').replace('endseq', '')
        return 'Caption generation failed'
    except Exception as e:
        return str(e)

@app.post('/generate_caption')
async def generate_caption_route(file: UploadFile = File(...)) -> dict:
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        caption = generate_caption(image, model, tokenizer, max_length)
        return {'caption': caption}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)