from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.mixture import GMM
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt



def get_different_classes(imgs, labels):
	d = {'1' : np.empty((0,1024)),'2': np.empty((0,1024)), '3': np.empty((0,1024)), '4': np.empty((0,1024)), '5': np.empty((0,1024)), '6': np.empty((0,1024)), '7': np.empty((0,1024))}

	imgs, labels, train_scaler = get_input(imgs, labels)

	for i in range(len(labels)):
		d[str(labels[i])] = np.vstack([d[str(labels[i])], imgs[i]])
	
	return d

def get_input(imgs, labels, scaler=None):
    I = np.rollaxis(imgs, 2)
    I = np.reshape(I, (I.shape[0], -1))

    I = np.asarray(I, dtype=np.float64)
    if not scaler:
        scaler = StandardScaler(copy=False)
    
    I = scaler.fit_transform(I)
    L = np.ravel(labels)
    
    return I, L, scaler


def get_input_pca(imgs, labels, pca=None):
    I = np.rollaxis(imgs, 2)
    I = np.reshape(I, (I.shape[0], -1))

    if not pca:
    	pca = RandomizedPCA(n_components=None, copy=False, iterated_power=3, whiten=False)
    
    I = pca.fit_transform(I)
    L = np.ravel(labels)

    return I, L, pca


def learn_knn(training_input, training_labels, nn):
    knn = KNeighborsClassifier(n_neighbors=nn, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)
    knn.fit(training_input, training_labels)
    return knn


def learn_nn(training_input, training_labels):
    # TODO: replace with sknn
    nn = MLPClassifier(hidden_layer_sizes=(100, ), activation='logistic', algorithm='adam', alpha=0.0001,
                       batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200,
                       shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                       nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                       epsilon=1e-08)
    nn.fit(training_input, training_labels)
    return nn


def learn_log_reg(training_input, training_labels):
    lr = LogisticRegressionCV(cv=5, penalty='l2', solver='lbfgs', tol=0.0001, max_iter=100, class_weight='balanced',
                              n_jobs=-1, multi_class='multinomial')
    lr.fit(training_input, training_labels)
    return lr


def learn_lin_reg(training_input, training_labels):
    lr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=-1)
    lr.fit(training_input, training_labels)
    return lr


def learn_svm(training_input, training_labels):
    # TODO: narrow down to necessary params
    svm = SVC(C=1.0, kernel='poly', degree=7, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001,
              cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None,
              random_state=None)
    svm.fit(training_input, training_labels)
    return svm


def learn_lin_svm(training_input, training_labels):
    svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=0.0001, multi_class='ovr', fit_intercept=True,
                    class_weight='balanced', max_iter=1000)
    svm.fit(training_input, training_labels)
    return svm

def learn_gmm(training_input):
	gmm = GMM(n_components=1, covariance_type='tied', random_state=None, thresh=None, tol=0.0001, min_covar=0.001, n_iter=100, n_init=1, params='wmc', init_params='wmc', verbose=0)
    gmm.fit(training_input)
	return gmm

def calculate_accuracy(model, test_imgs, test_labels):
    return model.score(test_imgs, test_labels)

def classify(model, test_img, emotions):
    return emotions[model.predict(test_img)[0]]

def get_accuracy(model, test_input, test_targets):
	num_examples = len(test_targets)
	ctr = 0

	for i in range(num_examples):
		if (round(model.predict(test_input[i].reshape(1,-1))[0]) == test_targets[i]):
			ctr += 1
	
	return (ctr/float(num_examples))

def plot(x_data, y_data, x_label, y_label, fig_name):
	fig = plt.plot(x_data,y_data, 'r-o')
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.savefig(fig_name)
	plt.show()

if __name__ == '__main__':
    training_data = sio.loadmat('labeled_images.mat')

    # ith image is training_imgs[:,:,i], and its label is emotions[training_labels[i][0]]
    training_imgs = training_data['tr_images']
    training_labels = training_data['tr_labels']

    emotions = {1: 'Anger', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Sad', 6: 'Surprise', 7: 'Neutral'}

#     d = get_different_classes(training_imgs, training_labels)
#         
# 
#     valid_data = np.empty((0,1024))
#     test_data = np.empty((0,1024))
#     valid_labels = np.empty((0,1))
#     tr_labels = np.empty((0,1))
#     tr_data = np.empty((0,1024))
# 
# 	for k in ['1', '2', '3', '4', '5', '6', '7']:   # has to be ordered as d.keys() wouldnt be ordered and messes up
# 		l = (d[k].shape[0] / 10) * -1
# 
# 		v = d[k][l:, :]
# 		t = d[k][2 * l:l, :]
# 		d[k] = d[k][:2 * l, :]
# 
# 		valid_data = np.vstack((valid_data, v))
# 		labels = np.ones((abs(l), 1)) * int(k)
# 		valid_labels = np.vstack((valid_labels, labels))   # will be same as test labels
# 
# 		test_data = np.vstack((test_data, t))
# 
# 		tr_data = np.vstack((tr_data, d[k]))
# 		t_labels = np.ones((d[k].shape[0], 1)) * int(k)
# 		tr_labels = np.vstack((tr_labels, t_labels))
# 
# 
#     data = {'tr_data': tr_data, 'valid_data': valid_data, 'test_data': test_data, 'valid_labels': np.ravel(valid_labels), 'tr_labels': np.ravel(tr_labels)}
# 
#     sio.savemat('data.mat', data)
#     
#     sio.savemat('divided_tr_data.mat', d)
    # train_imgs = training_imgs[:,:,:2500]
    # train_labels = training_labels[:2500]
    # test_imgs = training_imgs[:,:,:2500]
    # test_labels = training_labels[:2500]

    # train_input, train_targets, train_scaler = get_input(train_imgs, train_labels)
    # test_input, test_targets, _ = get_input(test_imgs, test_labels, train_scaler)

    # train_input_pca, train_targets_pca, train_pca = get_input_pca(train_imgs, train_labels)
    # test_input_pca, test_targets_pca, _ = get_input(test_imgs, test_labels, train_pca)

    # knn = learn_knn(train_input, train_targets, 5)
    # print calculate_accuracy(knn, test_input, test_targets)

    # nn = learn_nn(train_input, train_targets)
    # print calculate_accuracy(nn, test_input, test_targets)

    # log_reg = learn_log_reg(train_input, train_targets)
    # print calculate_accuracy(log_reg, test_input, test_targets)

    # log_reg_pca = learn_log_reg(train_input_pca, train_targets_pca)
    # print calculate_accuracy(log_reg_pca, test_input_pca, test_targets_pca)
    # print get_accuracy(log_reg_pca, test_input_pca, test_targets_pca)

    # lin_reg = learn_lin_reg(train_input, train_targets) 
    # print calculate_accuracy(lin_reg, test_input, test_targets) # 72% accuracy
    # print get_accuracy(lin_reg, test_input, test_targets)   # doesn't work as expected for this because predictionsare floats

    # f = sio.loadmat('data.mat')
    # log_reg = learn_log_reg(f['tr_data'], f['tr_labels'].T)
    # print calculate_accuracy(log_reg, f['valid_data'], f['valid_labels'].T)   


	

    #print get_accuracy(gmm, f['valid_data'], f['valid_labels'].T) 
    
    # lin_reg = learn_lin_reg(train_input, train_targets)
    # print calculate_accuracy(lin_reg, test_input, test_targets)

    # lin_svm = learn_lin_svm(train_input, train_targets)
    # print calculate_accuracy(lin_svm, test_input, test_targets)

    # svm = learn_svm(train_input, train_targets)
    # print calculate_accuracy(svm, test_input, test_targets)
 	
 	# pub_test = sio.loadmat('public_test_images')['public_test_images']
    # pub_test_input, pub_test_target, _ = get_input(pub_test, test_labels, train_scaler)
    # produce_submission(log_reg_pca, pub_test_input)



    scaler = StandardScaler(copy=False)
    pca = RandomizedPCA(n_components=200, copy=False, iterated_power=3, whiten=False)
    
	f = sio.loadmat('data.mat')
	d = sio.loadmat('divided_tr_data.mat')

	log_prob = np.empty((290,7))
	
	gmms = {'1' : None, '2' : None, '3' : None, '4' : None, '5' : None, '6' : None, '7' : None}
	
	for k in ['1', '2', '3', '4', '5', '6', '7']:
		gmms[k] = learn_gmm(pca.fit_transform(d[k]))
		probs = gmms[k].score(pca.fit_transform(f['valid_data']))
		log_prob[:,int(k) - 1] = probs
    
	pred = np.argmax(log_prob, axis=1) + 1 # classes are 1-7, not 0-6
	print np.mean(pred == f['valid_labels'][0])