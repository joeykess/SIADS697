FROM continuumio/miniconda3

COPY requirements-docker.txt /tmp/
COPY ./apps /apps
COPY ./assets /assets
COPY index.py ./
COPY app.py ./
COPY financial_metrics.py ./
COPY port_charts.py ./
COPY model_descriptions.py ./

RUN pip install -r /tmp/requirements-docker.txt

EXPOSE 8050

ENTRYPOINT [ "python3" ]
CMD [ "index.py" ]