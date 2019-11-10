# Keras Docker & Sagemaker Example

This example was modified from:
https://medium.com/@richardchen_81235/custom-keras-model-in-sagemaker-277a2831ac67

Original Github Repo:
https://github.com/rca241231/sagemaker_example

This example uses the pima indians dataset to detect diabetes. 

## Basic Overview

* build_and_push.sh - creates/updates/pushes the docker image.
* pima/train - sets the locations of data/models in container. Launches
* pima/serve - launches the gunicorn(middleware) and nginx(exposed webserver) 
processes. gunicorn is able to interface with our flask app.
* predictor.py - Our flask app that grabs the model and computes 
the predictions
* sage_train_local.py - train & deploy locally
* sage_train_cloud.py - train & deploy in sagemaker
* sage_get_predictions.py - test against locally deployed endpoint. 


## Usage
To update the container with modified code or to run the dockerfile:

``` shell
/bin/bash update_image.sh
```

To train and launch model locally:
``` shell
python sage_train_local.py
```
To train and launch model in AWS:
``` shell
python sage_train_cloud.py
```