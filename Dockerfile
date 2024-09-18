FROM python:3.9-slim


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libzbar-dev \
    libgl1-mesa-glx \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html

RUN pip install opencv-python-headless

## For cuda based, https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

COPY requirements.txt .


RUN pip install -r requirements.txt


COPY . /app
WORKDIR /app


CMD ["python", "main.py"]

