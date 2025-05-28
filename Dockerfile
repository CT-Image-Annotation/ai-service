FROM python:3.13-slim

# Install dependencies needed for conda and your packages
RUN apt-get update && apt-get install -y wget bzip2 && rm -rf /var/lib/apt/lists/*

# Install Miniconda silently
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /app
COPY environment.yml .

RUN conda update -n base -c defaults conda -y && \
    conda env create -f environment.yml && \
    conda clean -a -y

SHELL ["conda", "run", "-n", "medsam2", "/bin/bash", "-c"]

COPY . .

EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "medsam2", "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
