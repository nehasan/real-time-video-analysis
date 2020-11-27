import cv2
import numpy as np
import time

from flask import Flask, render_template, Response

app = Flask(__name__)





@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')
    
    #return str(classes)

cap=cv2.VideoCapture(0) #0 for 1st webcam


def geny():
    #Load YOLO
    net = cv2.dnn.readNet("yolov4-tiny.weights","yolov4-tiny.cfg") # Original yolov3
    #net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo
    classes = []
    with open("coco.names","r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    colors= np.random.uniform(0,255,size=(len(classes),3))

    #loading image
    # cap=cv2.VideoCapture("V1.mp4") #0 for 1st webcam
    
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time= time.time()
    frame_id = 0

    while True:
        _,frame= cap.read() # 

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5) 

        frame_id+=1
        
        height,width,channels = frame.shape
        #detecting objects
        blob = cv2.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False) #reduce 416 to 320    

            
        net.setInput(blob)
        outs = net.forward(outputlayers)
        #print(outs[1])


        #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids=[]
        confidences=[]
        boxes=[]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    #onject detected
                    center_x= int(detection[0]*width)
                    center_y= int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)

                    #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    #rectangle co-ordinaters
                    x=int(center_x - w/2)
                    y=int(center_y - h/2)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x,y,w,h]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)


        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence= confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
                cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)
                

        elapsed_time = time.time() - starting_time
        fps=frame_id/elapsed_time
        cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
        
        #cv2.imshow("Image",frame)

        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
        
        if key == 27: #esc key stops the process
            break;
        
    # cap.release()    
    # cv2.destroyAllWindows()



def gen():
    """Video streaming generator function."""
    # cap = cv2.VideoCapture("V1.mp4")

    # Read until video is completed
    while(cap.isOpened()):
      # Capture frame-by-frame
        ret, img = cap.read()
        if ret == True:
            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        else: 
            break



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_overlay')
def video_overlay():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(geny(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
   app.run()

    

