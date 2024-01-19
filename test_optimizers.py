from logreg_train import OPTIMIZERS
from utils import OPTIMIZER_CHOICES, OPTIMIZER_DEFAULT


def test_same():
    assert sorted(OPTIMIZER_CHOICES) == sorted(list(OPTIMIZERS.keys()))


def test_in():
    assert OPTIMIZER_DEFAULT in OPTIMIZER_CHOICES
