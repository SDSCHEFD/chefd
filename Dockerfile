FROM ubuntu:14.04

# Install.

RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y python-dev && \
  apt-get install -y python-pip && \
  
 


# Define working directory.
WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt
# Define default command.
CMD ["python", "test.py"]
