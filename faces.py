import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {"person_name":1}
with open("lables.pickle","rb") as f:
	orginal_label = pickle.load(f)
	lables = {v:k for k,v in orginal_label.items()} #key value pairs


cap = cv2.VideoCapture(0)

while(True):

	#capture frame by frame
	ret , frame = cap.read()

	#covert to gray for face recognition
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for(x,y,w,h) in faces:
		#print(x,y,w,h)
		#region of iterest
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]


		#recognize frame
		#id & confidence
		id_, conf = recognizer.predict(roi_gray)
		if conf >=15 and conf <= 85:
			print(id_)
			print(lables[id_])

			font = cv2.FONT_HERSHEY_SIMPLEX
			name = lables
			color = (255,255,255)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
		img_item = "my-img.png"
		cv2.imwrite(img_item, roi_gray)

		#draw a rectangle
		color = (255,0,0) #bluegreenred BGR 0-255
		stroke = 2
		end_cord_x = x+w #width
		end_cord_y = y+h #height
		cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y),color,stroke)



	#dsiplay the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break


#release capture
cap.release()
cv2.destroyAllWindows()