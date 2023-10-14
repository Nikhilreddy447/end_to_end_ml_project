FROM python:3.8-alpine
COPY ./ app
WORKDIR /app
RUN apk add --no-cache build-base
RUN pip install -r requirements.txt
CMD python app.py

