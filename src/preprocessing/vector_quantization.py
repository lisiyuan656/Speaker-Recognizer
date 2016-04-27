import numpy

class VectorQuantization:
    len_feature = 13

    def __init__(self):
        self.class_names = []
        self.train_datasets = numpy.ndarray(shape = (0, VectorQuantization.len_feature), dtype = numpy.float64)
        self.train_labels = numpy.ndarray(shape = (0), dtype=numpy.int)

    def train(self, class_names, train_datasets, train_labels):
        self.class_names = class_names;
        self.train_datasets = train_datasets
        self.train_labels = train_labels

    def predict(self, test_data):
        scores = range(len(self.class_names))
        for i in range(len(self.class_names)):
            scores[i] = 0;
        for k in range(test_data.shape[0]):
            distance = [numpy.sum((self.train_datasets[j]-test_data[k])**2) for j in range(self.train_labels.shape[0])]
            for i in range(len(self.class_names)):
                distance_per_label = [distance[j] for j in range(self.train_labels.shape[0]) if self.train_labels[j]==i]
                scores[i] += min(distance_per_label)
        print ('the person is: ', self.class_names[scores.index(min(scores))])
        return scores.index(min(scores))

    def test(self, test_dataset_list, test_label_list):
        success = 0
        for i in range(len(test_dataset_list)):
            if self.predict(test_dataset_list[i])==test_label_list[i]:
                success += 1
                print(success)
        print ('the accuracy of the algorithm is: ', success*1.0/len(test_dataset_list))
