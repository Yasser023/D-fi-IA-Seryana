FROM python:3.10
ADD requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ADD features_hotels.csv .
ADD test_set.csv .
ADD trainset.csv .


ADD train.py .
ADD app.py .

CMD [ "python3" , "./app.py" ]