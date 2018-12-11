FROM continuumio/miniconda3

ADD environment.yml /tmp/environment.yml
RUN conda env update -n base -f /tmp/environment.yml

ADD . .

RUN python train.py

EXPOSE 8080
CMD ["python", "serve.py"]
