from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
import numpy as np


def get_input_old(imgs, labels):
    s = (1, 1024)   #each image will be a 1024 length vector
    I = np.zeros(s)
    L = np.zeros((1,1))

    for i in range(imgs.shape[2]):
        im = imgs[:,:,i].reshape(1, 1024)
        I = np.vstack((I, im))    			#add the image to the matrix as the last row of the matrix
        L = np.vstack((L, labels[i][0]))    #add the image to the matrix as the last row of the matrix

    I = np.delete(I, (0), axis=0)
    I = np.delete(L, (0), axis=0)

    return I, L


def get_input(imgs, labels):
    I = np.rollaxis(imgs, 2)
    I = np.reshape(I, (I.shape[0], -1))
    L = np.ravel(labels)
    return I, L


def learn_knn(training_input, training_labels, nn):
    knn = KNeighborsClassifier(n_neighbors=nn, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)
    knn.fit(training_input, training_labels)
    return knn


def learn_nn(training_input, training_labels):
    # TODO: narrow down to necessary params
    nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', algorithm='adam', alpha=0.0001,
                       batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
                       shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
    nn.fit(training_input, training_labels)
    return nn


def learn_lr(training_input, training_labels):
    # TODO(shayousefi): play with tol and max_iter for better accuracy
    lr = LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', tol=0.0001, max_iter=100, class_weight='balanced',
                              n_jobs=-1, multi_class='multinomial')
    lr.fit(training_input, training_labels)
    return lr


def calculate_accuracy(model, test_imgs, test_labels):
    return model.score(test_imgs, test_labels)


def classify(model, test_img, emotions):
    return emotions[model.predict(test_img)[0]]


if __name__ == '__main__':
    training_data = sio.loadmat('labeled_images.mat')

    # ith image is training_imgs[:,:,i], and its label is emotions[training_labels[i][0]]
    training_imgs = training_data['tr_images']
    training_labels = training_data['tr_labels']

    emotions = {1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Surprise', 7: 'Neutral'}

    train_input, train_labels = get_input(training_imgs[:,:,:2500], training_labels[:2500])
    test_input, test_labels = get_input(training_imgs[:,:,2500:], training_labels[2500:])

    # knn = learn_knn(train_input, train_labels, 5)
    # print calculate_accuracy(knn, test_input, test_labels)

    # nn = learn_nn(train_input, train_labels)
    # print calculate_accuracy(nn, test_input, test_labels)

    lr = learn_lr(train_input, train_labels)
    print calculate_accuracy(lr, test_input, test_labels)
