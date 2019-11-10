from sagemaker import get_execution_role

role = get_execution_role()


from sagemaker.estimator import Estimator

hyperparameters = {'train-steps': 100}

instance_type = 'ml.p2.xlarge'

estimator = Estimator(role=role,
                      train_instance_count=1,
                      train_instance_type=instance_type,
                      image_name = '<insert image name>',
                      output_path='s3://<insert output path>',
                      hyperparameters=hyperparameters)


# The fit function uploads the tar.gz file to the bucket.
# fit checks to see if this file is actually there, but it may not be using it.
estimator.fit('s3://<insert bucket path to csv>')
# estimator.fit('file:///tmp/cifar-10-data')
# estimator.fit('file://./data/pima-indians-diabetes.csv')

# Code - https://github.com/awslabs/amazon-sagemaker-examples/blob/master/advanced_functionality/tensorflow_bring_your_own/tensorflow_bring_your_own.ipynb
predictor = estimator.deploy(1, instance_type)





