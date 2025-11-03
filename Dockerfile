FROM python:3.12.7

WORKDIR /MathRecognition
COPY ./Docker/requirements.txt .
COPY ./backend/Main.py ./Main.py
COPY ./backend/Recognition.py ./Recognition.py
COPY ./backend/saved_model ./backend/saved_model
COPY ./encoder_data ./encoder_data


RUN apt-get update && apt-get install -y libgl1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "./Main.py"]
