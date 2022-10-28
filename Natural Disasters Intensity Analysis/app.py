from flask import Flask,render_template,request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
app = Flask(__name__)
model=load_model('disaster.h5')
@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('home.html')
@app.route('/predict', methods=['GET'])
def index1():
    # Main page
    return render_template('launch.html')
@app.route('/intro', methods=['GET'])
def index2():
    # Main page
    return render_template('intro.html')
@app.route('/videopredict', methods=['GET', 'POST'])
def running():
    webcam = cv2.VideoCapture(0)
    #while True:
       # _,frame=webcam.read()
        #frame=cv2.flip(frame,1)
    while True:
        (grabbed,frame)=webcam.read()
        if(not grabbed):
            break
        W = webcam.get(cv2. CAP_PROP_FRAME_WIDTH )
        H = webcam.get(cv2. CAP_PROP_FRAME_HEIGHT )
        if(W is None and H is None):
            (H,W)=frame.shape[:2]
        output = frame.copy()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frame=cv2.resize(frame,(64,64))
        x=np.expand_dims(frame,axis=0)
        result=np.argmax(model.predict(x),axis=1)
        index=["Cyclone","EarthQuake","Flood","WildFire"]
        result=str(index[result[0]])
        cv2.putText(output,"activity : {}".format(result),(10,120),cv2.FONT_HERSHEY_PLAIN,1,(0,255,255),1)
        cv2.imshow("Output",output)
        key=cv2.waitKey(1)& 0xFF
        if(key==ord('q')):
            break
    print('[INFO] cleaning')
    webcam.release()
    cv2.destroyAllWindows()
    return render_template("home.html")




if __name__ == '__main__':
    app.run(debug=True)