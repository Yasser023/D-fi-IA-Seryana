FROM python:3.10
ADD requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ADD features_hotels.csv .
ADD sample_submission.csv .
ADD test_set.csv .
ADD dataset_1.csv .
ADD dataset_2.csv .
ADD dataset_3.csv .
ADD dataset_4.csv .
ADD dataset_5.csv .
ADD dataset_6.csv .
ADD dataset_1000_1.csv . 
ADD dataset_1000_2.csv .
ADD dataset_1000_3.csv . 
ADD dataset_v2_1.csv . 
ADD dataset_v2_2.csv . 


ADD train.py .
ADD app.py .

CMD [ "python3" , "./app.py" ]