FROM alpineintuition/archipel-base-cpu:latest

ARG FACE_PIXELIZER=/opt/face_pixelizer
RUN mkdir ${FACE_PIXELIZER}

COPY face-pixelizer/ ${FACE_PIXELIZER}

RUN cd ${FACE_PIXELIZER}

RUN pip install -r requirements.txt

ENV PYTHONPATH="${FACE_PIXELIZER}:${PYTHONPATH}"
