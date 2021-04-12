import face_recognition
import cv2
import argparse
import os
import numpy as np
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="img")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--source', type=str, default=None )
    parser.add_argument('--task', type=str, default='display') # sort, convert
    return parser.parse_args()

def distance(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])*(pt1[0]-pt2[0]) + (pt1[1]-pt2[1])*(pt1[1]-pt2[1]))

def pose(file):
    f = feature(file)
    if f == None:
        print("no feature", file)
        return None
    leb, reb, nose, chin = f[0], f[1], f[2], f[3]    
    ld = 0.5*(distance(leb[0],nose[0])+distance(leb[-1],nose[0]))
    rd = 0.5*(distance(reb[0],nose[0])+distance(reb[-1],nose[0])) 
    return ld-rd

def sort(files, input, output):
    d = len(files)*[0.0]
    for i in range(0, len(files)):
        d[i] = pose(os.path.join(input, files[i]))
    sd = d.copy()
    sd.sort()
    # print(d)
    # print(sd)
    for i in range(0,len(sd)):
        index, = np.where(np.isclose(d,sd[i]))
        # print(index)
        for j in range(0,len(index)):
            source = cv2.imread(os.path.join(input, files[index[j]]))
            cv2.imwrite(os.path.join(output, str(i)+'_'+str(j)+'.jpg'), source)

def feature(file):
    image = face_recognition.load_image_file(file)
    face_landmarks_list = face_recognition.face_landmarks(image)
    if len(face_landmarks_list) == 0:
        print("no feature found")
        return None

    face_landmarks = face_landmarks_list[0]
    image = cv2.polylines(image, np.array(face_landmarks["left_eyebrow"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["right_eyebrow"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["left_eye"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["right_eye"]).reshape((-1,1,2)), 1, (0,0,255,2))
    image = cv2.polylines(image, np.array(face_landmarks["top_lip"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["bottom_lip"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["nose_bridge"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["nose_tip"]).reshape((-1,1,2)), 1, (0,0,255),2)
    image = cv2.polylines(image, np.array(face_landmarks["chin"]).reshape((-1,1,2)), 1, (0,0,255),2)
    cv2.namedWindow("Akira", cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('Akira', image)
    cv2.waitKey(3)

    left_eb = face_landmarks["left_eyebrow"]
    right_eb = face_landmarks["right_eyebrow"]
    nose = face_landmarks["nose_bridge"]
    chin = face_landmarks["chin"]

    return (left_eb, right_eb, nose, chin)
    
def process(file, encode):
    image = face_recognition.load_image_file(file)
    # print(face_landmarks_list)
    (h, w) = image.shape[:2]
    #image = cv2.resize(image, None, fx=0.5, fy=0.5)
    face_locations = face_recognition.face_locations(image)

    index = 0
    if encode != None:
        unknown_encode = face_recognition.face_encodings(image)
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

        f = feature(file)
        nose = f[2]
        ref0,ref1 = nose[0],nose[-1]
        fh = 3*int(distance(ref0,ref1))
        fw = 2*fh
        # print(ref0, ref1)
        
        t1 = ref0[1] - 2*fh # u, height
        b1 = ref0[1] + 5*fh 
        r1 = ref0[0] + 2*fw # v, width
        l1 = ref0[0] - 2*fw
        
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
    task = args.task 
    path = os.walk(args.input)

    if task == 'convert':
        known_encode = None
        if args.source != None:
            known_image = face_recognition.load_image_file(args.source)
            known_encode = face_recognition.face_encodings(known_image)[0]

        for root, directories, files in path:
            for directory in directories:
                print(directory)
            for file in files:
                inputname = os.path.join(args.input, file) 
                print(inputname)
                img = process(inputname, known_encode)
                cv2.imwrite(os.path.join(args.output, file), img)
    elif task == 'display':
        for root, directories, files in path:
            for directory in directories:
                print(directory)
            for file in files:
                inputname = os.path.join(args.input, file) 
                print(inputname)
                feature(inputname)
    elif task == 'sort':
        for root, directories, files in path:
            sort(files,args.input,args.output)
