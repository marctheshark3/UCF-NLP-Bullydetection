FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN pip3 install pandas>=0.24.1 nltk>=3.4 scikit-learn>=0.20.3
RUN python3 -m nltk.downloader -d /usr/share/nltk_data all

RUN mkdir -p /bullydetect

WORKDIR /bullydetect
