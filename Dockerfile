FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 기본 패키지
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev python3-venv \
    git wget curl nano vim \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    software-properties-common \
    lsb-release gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# ROS Noetic 설치
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-sensor-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-std-msgs \
    ros-noetic-tf \
    ros-noetic-tf2-ros \
    ros-noetic-cv-bridge \
    ros-noetic-rosbridge-server \
    python3-rospkg \
    python3-catkin-pkg \
    python3-empy \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python3 -m pip install --upgrade pip setuptools wheel

# PyYAML 충돌 방지
# Ubuntu/ROS가 설치한 PyYAML 5.3.1을 제거하지 않고, pip 쪽 PyYAML을 우선 사용하게 함
RUN python3 -m pip install --no-cache-dir --ignore-installed PyYAML==6.0.1

# 일반 Python 패키지
RUN python3 -m pip install --no-cache-dir --retries 10 --timeout 120 \
    numpy==1.24.4 \
    scipy==1.10.1 \
    pandas==2.0.3 \
    matplotlib==3.7.5 \
    opencv-python==4.9.0.80 \
    tqdm \
    easydict \
    scikit-learn==1.3.2 \
    open3d==0.19.0

# PyTorch CUDA 11.8
# 대용량 다운로드라 마지막 단계로 분리함
RUN python3 -m pip install --no-cache-dir --retries 20 --timeout 300 \
    torch==2.1.2+cu118 \
    torchvision==0.16.2+cu118 \
    torchaudio==2.1.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

WORKDIR /workspace/morai-3d-detection

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

CMD ["/bin/bash"]
