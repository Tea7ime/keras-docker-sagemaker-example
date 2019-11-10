
# For more information on creating a Dockerfile
# https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile

FROM continuumio/miniconda3
RUN conda create --name crw-model python=3.7
RUN echo "source activate crw-model" > ~/.bashrc

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
         gnupg2 \
         curl

# Download TensorFlow Serving
# https://www.tensorflow.org/serving/setup#installing_the_modelserver
RUN echo "deb [arch=amd64] http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list
#RUN curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -
RUN apt-get update && apt-get install tensorflow-model-server
RUN pip install sagemaker
RUN conda install -c conda-forge keras=2.2.5
RUN conda install scikit-learn gunicorn flask gevent pandas -y
ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
COPY /pima /opt/ml/code
WORKDIR /opt/ml/code