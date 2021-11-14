"""
The file is used for utils general or to be tested

"""


from .association_rules_eval import cal_support
from .association_rules_eval import cal_confidence
from .association_rules_eval import make_combinations

__all__ = ["cal_support", "cal_confidence", "make_combinations"]
