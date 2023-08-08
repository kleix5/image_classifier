FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx \
    && apt-get install -y libglib2.0-0

COPY __init__.py app.py model_util.py roi_matching.py

COPY /uploads /app/uploads

COPY /templates /app/templates 

COPY requirements.txt .

RUN chmod -R ugo+rwx /app/uploads

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["flask", "run", "--host", "0.0.0.0"]