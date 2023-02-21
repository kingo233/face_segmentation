FROM bvlc/caffe:gpu
# FROM bvlc/caffe:cpu


RUN apt-get -y update
RUN apt-get -y install python-tk
RUN pip install --upgrade pip
RUN pip install pathlib
RUN pip install opencv-python==4.2.0.32 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install numpy==1.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace/face_segmentation

ENTRYPOINT [ "python" , "face_seg.py" ]
