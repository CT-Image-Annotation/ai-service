FROM continuumio/miniconda3:latest

WORKDIR /app
COPY environment.yml .

RUN conda env create -f environment.yml && \
    conda clean -a -y

SHELL ["conda", "run", "-n", "medsam2", "/bin/bash", "-c"]

COPY . .
EXPOSE 5000

CMD ["conda", "run", "--no-capture-output", "-n", "medsam2", "gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
