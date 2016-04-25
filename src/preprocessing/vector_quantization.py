import numpy

class vector_quantization:

    def __init__(self):
        self.class_names = []
        len_feature = 13
        self.train_datasets = numpy.ndarray(shape = (0, len_feature), dtype = numpy.float64)
        self.train_labels = numpy.ndarray(shape = (0), dtype=numpy.int)
    def train(self, class_names, train_datasets, train_labels):
        self.class_names = class_names;
        self.train_datasets = train_datasets
        self.train_labels = train_labels
    def predict(self, test_data):
        scores = []
        for i in range(len(self.class_names)):
            distance = 0
            



    def test(self, class_names, test_datasets, test_labels):
