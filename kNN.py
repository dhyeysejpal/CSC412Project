from sklearn.neighbors import KNeighborsClassifier
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import scale

def get_input(imgs, labels):
    s = (1, 1024)   #each image will be a 1024 length vector
    I = np.zeros(s)
    L = np.array([])

    for i in range(imgs.shape[2]):
        im = imgs[:,:,i].reshape(1, 1024)
        I = np.vstack((I, im))    #add the image to the matrix as the last row of the matrix
        L = np.append(L, labels[i][0])    #add the image to the matrix as the last row of the matrix

    I = np.delete(I, (0), axis=0)
    I = scale(I)

    return I, L

def learn_knn(training_input, training_labels, nn):
    knn = KNeighborsClassifier(n_neighbors=nn, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)
    knn.fit(training_input, training_labels)
    return knn

def calculate_accuracy(model, test_imgs, test_labels):
    return model.score(test_imgs, test_labels)

def classify(model, test_img, emotions):
    return emotions[model.predict(test_img)[0]]

if __name__ == '__main__':
    training_data = sio.loadmat('C:\\Users\\dhyey_000\\Desktop\\CSC412\\CSC412Project\\labeled_images.mat')

    # ith image is training_imgs[:,:,i], and its label is emotions[training_labels[i][0]]
    training_imgs = training_data['tr_images']
    training_labels = training_data['tr_labels']

    emotions = {1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Surprise', 7: 'Neutral'}
    
    train_input, train_labels = get_input(training_imgs[:,:,:2500], training_labels[:2500])
    test_input, test_labels = get_input(training_imgs[:,:,2500:], training_labels[2500:])
    
    
    knn = learn_knn(train_input, train_labels, 11)
    print calculate_accuracy(knn, test_input, test_labels)   