
logger = logging.getLogger(__name__)


class Metric:
    def __init__(self):
        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)

    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        return sum(self._tps.values()) if class_name is None else self._tps[class_name]

    def get_fp(self, class_name=None):
        return sum(self._fps.values()) if class_name is None else self._fps[class_name]

    def get_tn(self, class_name=None):
        return sum(self._tns.values()) if class_name is None else self._tns[class_name]

    def get_fn(self, class_name=None):
        return sum(self._fns.values()) if class_name is None else self._fns[class_name]

    def f_score(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        return precision, recall, f1

    def accuracy(self, class_name=None):
        tp = self.get_tp(class_name)
        fp = self.get_fp(class_name)
        fn = self.get_fn(class_name)
        return tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0

    def micro_avg_f_score(self):
        return self.f_score()[-1]

    def macro_avg_f_score(self):
        scores = [self.f_score(c)[-1] for c in self.get_classes()]
        return sum(scores) / len(scores) if len(scores) > 0 else 0.0

    def micro_avg_accuracy(self):
        return self.accuracy()

    def macro_avg_accuracy(self):
        accuracies = [self.accuracy(c) for c in self.get_classes()]
        return sum(accuracies) / len(accuracies) if len(accuracies) > 0 else 0.0

    def get_classes(self):
        all_classes = set(list(self._tps.keys()) + list(self._fps.keys()) + list(self._tns.keys()) + list(self._fns.keys()))
        return sorted([c for c in all_classes if c is not None])

    def to_dict(self):
        result = {}
        for n in self.get_classes():
            result[n] = {"tp": self.get_tp(n), "fp": self.get_fp(n), "fn": self.get_fn(n), "tn": self.get_tn(n)}
            result[n]["p"], result[n]["r"], result[n]["f"] = self.f_score(n)
        result["overall"] = {"tp": self.get_tp(), "fp": self.get_fp(), "fn": self.get_fn(), "tn": self.get_tn()}
        result["overall"]["p"], result["overall"]["r"], result["overall"]["f"] = self.f_score()
        return result
