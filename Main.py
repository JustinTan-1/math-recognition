import torch
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import json
from Recognition import Recognition2
from sklearn.preprocessing import LabelEncoder


app = FastAPI()

# Load saved model and encoder values
model = Recognition2()
model.load_state_dict(torch.load("./saved_model/model.pt", weights_only=True))

encoder = LabelEncoder()
encoder.classes_ = np.load('./encoder_data/classes.npy')


#For debugging purposes, uncomment all the cv2.imshow() methods

@app.post("/")
async def recieve_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert()
    #image.show()
    np_image = np.array(image)
    #cv2.imshow("Image", np_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    prediction = predict_symbol(np_image)
    prediction = json.dumps(prediction)
    return {"prediction": prediction}


def process_drawing(screen):
    gray = np.dot(screen[..., :3], [0.2989, 0.587, 0.114])
    #gray = np.transpose(gray, (1,0))
    #cv2.imshow("Image", gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #gray = cv2.resize(gray, (45,45), interpolation=cv2.INTER_AREA)
    gray = center_image(gray)
    if gray is None:
        return None
    gray = gray.astype(np.float32) / 255.0
    gray = (gray - 0.5) / 0.5
    tensor = torch.tensor(gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return tensor

def predict_symbol(screen):
    image = process_drawing(screen)
    if image is None:
        return None
    model.eval()
    with torch.no_grad():
        output = model(image)
        #_, prediction = torch.max(output, 1)
        _, sorted = torch.sort(output, 1, descending=True)
        prediction_array = []
        for i in range(0,3):
            prediction = encoder.inverse_transform([sorted[0][i].item()])
            prediction_array.append(prediction[0])
        print(prediction_array)
    return prediction_array

def center_image(image, size=45):
    ret,thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("Image", thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    coords = cv2.findNonZero(thresh)
    x,y,w,h = cv2.boundingRect(coords)
    if w == 0 or h == 0:
        return
    cropped = image[y:y+h, x:x+w]
    canvas = np.full((size, size),255, dtype=np.uint8)
    scale = min(size / w, size / h)
    resized = cv2.resize(cropped, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    x_offset = (size - resized.shape[1]) // 2
    y_offset = (size - resized.shape[0]) // 2
    canvas[y_offset:y_offset+resized.shape[0], x_offset:x_offset+resized.shape[1]] = resized
    #cv2.imshow("Image", canvas)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return canvas