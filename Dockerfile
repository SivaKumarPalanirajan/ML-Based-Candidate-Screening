FROM python:3.9:alpine
WORKDIR /app 
COPY . .
RUN pip install -r requirements.txt 
EXPOSE 8051 
cmd ["python","app.py"]