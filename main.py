from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from pymongo import ReturnDocument
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model
import json

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = load_model()
# def predict():
#     try:
#         string_tensor_filepath = tf.constant("bbaf2n.mpg")
#         video, annotations = load_data(tf.convert_to_tensor(string_tensor_filepath))
#         yhat = model.predict(tf.expand_dims(video, axis=0))
#         decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
#         converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
#         decoder_list = decoder.tolist()
#         return JSONResponse(content={"results": decoder_list, "text": converted_prediction}, status_code=200)
       
#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
async def home(request: Request):
    data = {"video": "../video.mp4"}
    return templates.TemplateResponse("index.html", {"request":request, "data": data})

@app.post("/lip-read")
async def lip_read():
    try:
        string_tensor_filepath = tf.constant("bbab9n.mpg")
        video, annotations = load_data(tf.convert_to_tensor(string_tensor_filepath))
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        converted_prediction =''.join([bytes.decode(x) for x in num_to_char(annotations.numpy()).numpy()])
        decoder_list = decoder.tolist()
        decoder_json = json.dumps(decoder_list)
        converted_json = json.dumps(converted_prediction)
        #return JSONResponse(content={"results": decoder_list, "text": converted_prediction})
        print(decoder_json,converted_json)
        data = {"sentence": converted_json, " array ": decoder_json}
        return templates.TemplateResponse("index.html", {"request":request, "data": data})
        # return JSONResponse(content={"array": decoder_list, "text": converted_prediction}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
