import csv
from collections import namedtuple, defaultdict
from configparser import ConfigParser

from sklearn import preprocessing
from sklearn.preprocessing import Normalizer
from sklearn.utils import resample


class DataSet:

    def __init__(self, cols=10):
        self.reservedColumns = cols     # reserved columns for dimension reduction
        self.cfg = ConfigParser()       # load configuration
        self.cfg.read('dataset.cfg')
        attrs, attrtypes = [], {}
        for i in self.cfg['content']['attributes'].split('.'):
            if i:
                attr, tp = i.split(':')
                attrs.append(attr.strip())
                attrtypes[attr] = tp.strip()
        self.record = namedtuple('record', attrs)
        self.fullData = []              # Raw data to process
        self.data = []                  # preserved data
        self.labelEncoders = {}         # label encoder
        self.labelSymbolSets = {}       # label type sets
        self.k = 0                      # hyper parameter k
        self.minClassThreshold = 10     # minimum number of samples in a class

    def load(self):
        """
        Load dataset from configuration file.
        :return: None
        """
        self.fullData, attrs, attrtypes = [], [], {}
        for i in self.cfg['content']['attributes'].split('.'):
            if i:
                attr, tp = i.split(':')
                attrs.append(attr.strip())
                attrtypes[attr.strip()] = tp.strip()
        for k, v in attrtypes.items():
            if v == 'symbolic':
                self.labelEncoders[k] = preprocessing.LabelEncoder()
                self.labelSymbolSets[k] = set()
        with open(self.cfg['file']['filepath'], newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            print("Loading dataset from ", self.cfg['file']['filepath'], "...")
            self.fullData = [self.record._make(i) for i in list(reader)]
            print("Success! Total ", len(self.fullData), " records.")

    def col_reduce(self, names=None):
        """
        Reduce attribute dimension via given column names.
        :param names: columns names to project on
        :return: extracted dataset list
        """

        cols = []
        for row in self.data:
            dic = row._asdict()
            row = [dic[name] for name in names]
            # parse from str representations
            row = self.parse(row, names)
            cols.append(row)
        # fit onehot encoder
        for idx, valueSet in self.labelSymbolSets.items():
            self.labelEncoders[idx].fit(list(valueSet))
        self.k = len(self.labelSymbolSets["label"])
        # second pass
        X, y = [], []
        for row in cols:
            for i in range(len(row)):
                if names[i] in self.labelEncoders:
                    row[i] = self.labelEncoders[names[i]].transform([row[i]])[0]
            X.append(row[:-1])
            y.append(row[-1])
        return X, y

    def col_reduce_default(self):
        """
        Using columns:
        as a defualt way of column reduction.
        :return: extracted dataset list
        """
        return self.col_reduce(names=self.cfg['content']['simple_attributes'].split(','))

    def parse(self, row, names):
        """
        Parsing list of strings and restore digits from literatures.
        :param row: list of string representation
        :param names: names of features
        :return: list of basic objects
        """
        # two pass: first pass collect symbolic values, second pass assign with numerical values
        for i in range(len(row)):
            if names[i] not in self.labelEncoders:
                row[i] = float(row[i])
            else:
                self.labelSymbolSets[names[i]].add(row[i])
        return row

    def sample(self, factor=0.001):
        """
        Resample full data to experimental data
        :param factor: portion of resampled data
        """
        self.data = resample(self.fullData, n_samples=int(len(self.fullData) * factor), replace=False, random_state=0)
        dic = defaultdict(int)
        for i in self.data:
            dic[i[-1]] += 1
        self.data = list(filter(lambda x: dic[x[-1]] > self.minClassThreshold, self.data))
        print("Sampling to ", len(self.data), " records...")

    @staticmethod
    def normalize(x):
        transformer = Normalizer().fit(x)
        return transformer.transform(x)

    @staticmethod
    def remap(y_test, y_train, labels):
        """
        Mapping test y data to predict labels.
        :param y_test: original test y data
        :param y_train: original full y data
        :param labels: true predict data
        :return: mapped test y data
        """
        dic = {k: v for k, v in zip(y_train, labels)}
        return [dic[i] for i in y_test]
