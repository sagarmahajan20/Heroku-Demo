import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask import request
import flask

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    url = request.args.get('url') 
    return {'object': "Car detected please stay alert", "url" : "CAr"}

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return 'Car Detected at 5 metre'

@app.route('/url/', methods=['GET']) #<string:url>
def get_task():
	url = request.args.get('url') 
    person_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

    #url = "https://maps.mapmyindia.com/place/P0015261904.jpg"

    # METHOD #1: OpenCV, NumPy, and urllib
    def url_to_image(url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # return the image
        return image
    def DetectPerson(img):
    # Read first frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Pass frame to our car classifier
    persons = person_classifier.detectMultiScale(gray, scaleFactor=1.106, minNeighbors=2,minSize=(32, 32))
        
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in persons:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    Txt="Person is detected"
    img = url_to_image(url)  
    DetectionText = DetectPerson(img)  

	return {'object': "object found at 5 meters", "url" : label}


if __name__ == "__main__":
    app.run(debug=True)
