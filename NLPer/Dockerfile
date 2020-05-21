FROM python:3.7-slim

MAINTAINER Wojciech Ozimek

RUN apt-get update -q && apt-get install -q -y \
    --no-install-recommends \
    curl \
    vim \
    && \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /nlper

ADD . /nlper

COPY scripts/entrypoint_test.sh /usr/local/bin/entrypoint_test.sh

RUN chmod +x /usr/local/bin/entrypoint_test.sh

RUN pip --no-cache-dir install -r requirements.txt

RUN pip install .

RUN curl -o /tmp/pl_spacy_model-0.1.0.tar.gz "http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=get&target=pl_spacy_model-0.1.0.tar.gz"

RUN tar -xvf /tmp/pl_spacy_model-0.1.0.tar.gz \
    && rm /tmp/pl_spacy_model-0.1.0.tar.gz

RUN cd pl_spacy_model-0.1.0 && pip install .

ENTRYPOINT ["entrypoint_test.sh"]

CMD ["/bin/bash"]