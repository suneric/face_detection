import face_recognition
import cv2
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default="img/437.jpg")
    parser.add_argument('--input', type=str, default="img")
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--source', type=str, default="akira.jpg" )
    return parser.parse_args()
    
def process(file,encode):
    image = face_recognition.load_image_file(file)
    (h, w) = image.shape[:2]
    #image = cv2.resize(image, None, fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(image)
    unknown_encode = face_recognition.face_encodings(image)

    index = 0
    for i in range(0,len(unknown_encode)):
        results = face_recognition.compare_faces([encode], unknown_encode[i])
        if results[0]:
            index = i
            break

    if len(face_locations) > index:
        # draw image with face recognition location
        (t,r,b,l) = face_locations[index]
        cv2.rectangle(image, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(image, "AKIRA", (l,t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.namedWindow("Akira", cv2.WND_PROP_FULLSCREEN)
        cv2.imshow('Akira', image)
        cv2.waitKey(3)

        fh = b-t
        fw = r-l
        t1 = t - 1.0*fh
        b1 = b + 3.0*fh 
        r1 = r + 1.25*fw
        l1 = l - 1.25*fw
        
        if t1 < 0:
            t1 = 0 
        if b1 > h:
            b1 = h
        if l1 < 0:
            l1 = 0 
        if r1 > w:
            r1 = w
        
        oh = 800
        origin = cv2.imread(file, cv2.IMREAD_COLOR)
        crop = origin[int(t1):int(b1), int(l1):int(r1)]
        (h, w) = crop.shape[:2]
        r = float(oh/h)
        ow = int(r*w)
        resize = cv2.resize(crop, (ow, oh))
        if ow > 700:
            resize = resize[0:799,int(0.5*ow)-350:int(0.5*ow)+350] 
        dst = cv2.blur(resize,(3,3))
        # dst = cv2.detailEnhance(resize, sigma_s=10, sigma_r=0.01)
        return dst
    else:
        return cv2.imread(file, cv2.IMREAD_COLOR)

        

if __name__ == "__main__":
    args = get_args()
    path = os.walk(args.input)
    known_image = face_recognition.load_image_file(args.source)
    known_encode = face_recognition.face_encodings(known_image)[0]
    for root, directories, files in path:
        for directory in directories:
            print(directory)
        for file in files:
            inputname = os.path.join(args.input, file) 
            print(inputname)
            img = process(inputname,known_encode)
            outputname = os.path.join(args.output, file) 
            cv2.imwrite(outputname, img)