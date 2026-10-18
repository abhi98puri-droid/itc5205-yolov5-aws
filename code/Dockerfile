# ITC5205 A2 - YOLOv5 detector packaged as a Lambda container image.
# A zip package cannot be used: PyTorch alone exceeds Lambda's 250 MB unzipped
# limit, while container images may be up to 10 GB.
# Python 3.12 base = Amazon Linux 2023 (glibc 2.34). The 3.11 base is Amazon
# Linux 2 (glibc 2.26), which cannot use newer manylinux_2_28 wheels, so pip
# fell back to compiling numpy from source and the build failed.
FROM public.ecr.aws/lambda/python:3.12

ENV YOLOV5_CONFIG_DIR=/tmp/Ultralytics \
    YOLOv5_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    PIP_NO_CACHE_DIR=1 \
    YOLO_DIR=/opt/yolov5 \
    YOLO_WEIGHTS=/opt/yolov5/yolov5s.pt

RUN dnf install -y git && dnf clean all

# CPU-only PyTorch; numpy<2 pinned in the same command so pip never picks numpy 2.x
RUN pip install "numpy<2" torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git /opt/yolov5 \
 && rm -rf /opt/yolov5/.git \
 && curl -sSL -o /opt/yolov5/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt \
 && chmod -R a+rX /opt/yolov5

RUN python -c "import torch; m = torch.hub.load('/opt/yolov5', 'custom', path='/opt/yolov5/yolov5s.pt', source='local', verbose=False); print('YOLOv5 OK -', len(m.names), 'classes')"

COPY app.py ${LAMBDA_TASK_ROOT}/
CMD ["app.handler"]
