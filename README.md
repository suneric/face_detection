# face_detection
Detect face in images

# Denedencies
- Anaconda
- Python 3
- Tensorflow 2
- face_recognition

# Start conda environment 
clone code then start environment with below commands
```
conda env create -f py3-tf2-cpu.yml
conda activiate py3-tf2-gpu
```

# Play
- display face feature in image
```
python face_in_image.py --task display --input [folder]
```

- convert image with specific face feature to a fix size image with appropriate resizing the image
```
python face_in_image.py --task convert --input [folder] --output [folder] --source [image with known face]
```

- sort images with face feature
```
python face_in_image.py --task sort --input [folder] --output [folder]
```