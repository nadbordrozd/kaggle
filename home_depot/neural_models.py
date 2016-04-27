from commons import logger, get_func_name, DATA_DIR

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils

def nn0(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

def nn1(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

def nn2(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

def nn3(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model
    
def nn4(input_dim):
    model = Sequential()
    model.add(Dense(32, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(32, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

def nn5(input_dim):
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(128, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model


def nn6(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model

def nn7(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model


def nn8(input_dim):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(10, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(64, input_shape=(input_dim,)))
    model.add(Activation('sigmoid'))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss='rmse', optimizer='adam')
    return model


class KerasWrapper(object):
    def __init__(self, model_maker):
        self.model_maker = model_maker
        self.model = None
        
    def fit(self, X, y):
        _, input_dim = X.shape
        self.model = self.model_maker(input_dim)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
       
    def __str__(self):
        return get_func_name(self.model_maker)
    
    

models = [lambda x=x: KerasWrapper(x) for x in [nn6, nn7, nn8]]