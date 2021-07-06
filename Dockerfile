FROM python:3.9.5

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV MLFLOW_TRACKING_URI=http://localhost:5000

EXPOSE 5000/tcp
EXPOSE 1234


COPY ./runServer.sh /usr/bin/
RUN chmod a+x /usr/bin/runServer.sh

COPY ./serveModel.sh /usr/bin/
RUN chmod a+x /usr/bin/serveModel.sh

CMD ["runServer.sh"]
