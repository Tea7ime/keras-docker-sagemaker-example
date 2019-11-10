from sagemaker import get_execution_role

# role = get_execution_role()
role = 'arn:aws:iam::078647912335:role/service-role/AmazonSageMaker-ExecutionRole-20191031T153899'

from sagemaker.estimator import Estimator

hyperparameters = {'train-steps': 100}

instance_type = 'ml.p2.xlarge'

estimator = Estimator(role=role,
                      train_instance_count=1,
                      train_instance_type=instance_type,
                      # image_name='sagemaker-pima-nn-test:latest',
                      image_name = '078647912335.dkr.ecr.us-west-2.amazonaws.com/sagemaker-pima-nn-test',
                      output_path='s3://spectra-sage/sagemaker/',
                      hyperparameters=hyperparameters)


# The fit function uploads the tar.gz file to the bucket.
# fit checks to see if this file is actually there, but it may not be using it.
estimator.fit('s3://spectra-sage/sage-nn-test/pima-indians-diabetes.csv')
# estimator.fit('file:///tmp/cifar-10-data')
# estimator.fit('file://./data/pima-indians-diabetes.csv')

# Code - https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb
predictor = estimator.deploy(1, instance_type)





