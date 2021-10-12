# Use python3.6
# See https://docs.docker.com/samples/library/python/ for latest 
FROM python:3.6

# Set it as the working directory
WORKDIR /lyu

ADD . /lyu/NLMwithRL

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
       python-opengl

# Upgrade pip3.6
RUN pip3.6 install --upgrade pip

# Move the requirements file into the image
COPY requirements.txt /tmp/

# Install the python requirements on the image
RUN pip3.6 install --trusted-host pypi.python.org -r /tmp/requirements.txt

ENV PATH /lyu/NLMwithRL/Jacinle/bin:${PATH}
RUN echo "export PATH=$PATH:/lyu/NLMwithRL/Jacinle/bin" >> /lyu/.bashrc

# Remove the requirements file - this is no longer needed
RUN rm /tmp/requirements.txt
RUN pip3.6 install ray

CMD ["/bin/bash"]

