# app/Dockerfile

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/* \
    || { cat /var/log/apt/term.log; exit 1; }


RUN git clone https://github.com/RiccardoDAndrea/LLM_RAG.git .

RUN git checkout docker

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "Prime-Chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]