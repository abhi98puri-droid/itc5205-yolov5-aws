# ITC5205 A2 - YOLOv5 detector packaged as a Lambda container image.
# A zip package cannot be used: PyTorch alone exceeds Lambda's 250 MB unzipped
# limit, while container images may be up to 10 GB.
FROM public.ecr.aws/lambda/python:3.11

# Lambda's filesystem is read-only except /tmp, so point every library that
# wants to write config/cache files there.
ENV YOLOV5_CONFIG_DIR=/tmp/Ultralytics \
    YOLOv5_AUTOINSTALL=false \
    MPLCONFIGDIR=/tmp/matplotlib \
    TORCH_HOME=/tmp/torch \
    PIP_NO_CACHE_DIR=1 \
    YOLO_DIR=/opt/yolov5 \
    YOLO_WEIGHTS=/opt/yolov5/yolov5s.pt

RUN yum install -y git && yum clean all && rm -rf /var/cache/yum

# CPU-only PyTorch (~200 MB instead of ~2.5 GB for the CUDA build)
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# YOLOv5 source (pinned release) + pretrained COCO weights baked into the image
# so nothing is downloaded at run time (faster, works without internet access).
RUN git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5.git /opt/yolov5 \
 && rm -rf /opt/yolov5/.git \
 && curl -sSL -o /opt/yolov5/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt \
 && chmod -R a+rX /opt/yolov5

# Build-time smoke test: the image fails to build if the model cannot load.
RUN python -c "import torch; m = torch.hub.load('/opt/yolov5', 'custom', path='/opt/yolov5/yolov5s.pt', source='local', verbose=False); print('YOLOv5 OK -', len(m.names), 'classes')"

COPY app.py ${LAMBDA_TASK_ROOT}/
CMD ["app.handler"]
