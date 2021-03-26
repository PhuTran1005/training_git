import cv2
import numpy as np # su lý mảng nhanh hơn nhờ cơ chế lưu data lien tuc

face_cascade = cv2.CascadeClassifier (cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
cap = cv2.VideoCapture (0) # Ham truy cap vao Webcam may tinh trong thu vien opencv

while (1): # Vong lap vo tan cho toi khi bam q de thoat vong lap
    ret, frame = cap.read () # frame la data ma lay duoc tu webcam; ret tra ve True neu truy cap vao webcam thanh cong

    # Lay du lieu train cho may, ta chuyen anh ve anh xam
    gray = cv2.cvtColor (frame, cv2.COLOR_BGR2GRAY)
    # cvt: convert to; frame: du lieu tu webcam cua may tinh
    # BGR_GRAY: Viet nguoc cua RGB to Gray
    faces = face_cascade.detectMultiScale (
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        )
    # thu vien nhan dien khuon mat mac dinh cua open cv
    for (x, y, w, h) in faces: # khoanh hinh vuong khuon mat
        # (x, y): toa do cua diem anh, tinh tien theo chieu ngang voi chiue doc de lay dc hinh vuong bao quanh mat
        cv2.rectangle (frame, (x, y), (x+w, y+h), (0, 225, 0), 5) # lay mau do
        # 2: do day cua hinh vuong
        # ham rectangle giup ve hinh vuong trong web cam
    cv2.imshow ('Nhan dien mat', frame)
    if (cv2.waitKey (1) & 0xFF == ord ('q')): # lenh ord ('q') giup thoat chuong trinh khi bam q
        break
cap.release ()  #giai phong bo nho
cv2.destroyAllWindows () # huy di
