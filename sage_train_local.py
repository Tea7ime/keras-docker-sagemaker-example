from sagemaker import get_execution_role

# you can also simply add your execution role here as a string.
role = get_execution_role()

from sagemaker.estimator import Estimator

hyperparameters = {'train-steps': 100}

instance_type = 'local'

estimator = Estimator(role=role,
train_instance_count=1,
train_instance_type=instance_type,
image_name='sagemaker-pima-nn-test:latest',
output_path='s3://<insert bucket path>',
hyperparameters=hyperparameters)

estimator.fit('file://./data/pima-indians-diabetes.csv')
# estimator.fit('s3://spectra-sage/sage-nn-test/pima-indians-diabetes.csv')

predictor = estimator.deploy(1, instance_type)


