FROM ubuntu:latest

RUN apt-get update && \
    apt-get install -y python3-pip

RUN update-ca-certificates && \
    pip3 config set global.trusted-host "pypi.org files.pythonhosted.org pypi.python.org"

RUN pip3 install umbridge

RUN pip3 install pyapprox

COPY minimal_server.py /

CMD python3 minimal_server.py
