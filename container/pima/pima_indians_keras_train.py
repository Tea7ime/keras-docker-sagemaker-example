# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import argparse
import os


# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, init= 'uniform' , activation= 'relu'))
    model.add(Dense(8, init= 'uniform' , activation= 'relu' ))
    # model.add(Dense(1, init= 'uniform' , activation= 'sigmoid'))
    model.add(Dense(1, init= 'uniform' , activation= 'sigmoid'))
    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

def train(model_dir, data_dir, train_steps):
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)
    # load pima indians dataset
    dataset_path = os.path.join(data_dir, 'pima-indians-diabetes.csv')
    # dataset_path = '../../data/pima-indians-diabetes.csv'
    dataset = numpy.loadtxt(dataset_path, delimiter=",", skiprows=1)
    # split into input (X) and output (Y) variables
    X = dataset[:,0:8]
    Y = dataset[:,8]
    # create model
    model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=1)
    # evaluate using 10-fold cross validation, this does not fit an actual model, it just evaluates.
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(model, X, Y, cv=kfold)
    # print(results)
    # print(results.mean())

    model.fit(X, Y)
    model.model.save(os.path.join(model_dir, 'model.h5'))
    print(f"Model summary after training:")
    print(model.model.summary())


def main(model_dir, data_dir, train_steps):
    # tf.logging.set_verbosity(tf.logging.INFO)
    train(model_dir, data_dir, train_steps)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    args = args_parser.parse_args()
    main(**vars(args))