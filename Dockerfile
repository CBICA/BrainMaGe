FROM continuumio/miniconda
WORKDIR /src
COPY . .
SHELL ["/bin/bash", "-c"]
RUN conda env create -f requirements.yml && conda init bash && source ~/.bashrc && conda activate brainmage && python setup.py install
ENV CONDA_DEFAULT_ENV=brainmage PATH=/opt/conda/envs/brainmage/bin:$PATH
ENTRYPOINT ["brain_mage_run"]
CMD -v
# docker build -t cbica/brainmage .