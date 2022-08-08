# Set of classes for validating that command line arguments do not conflict with one another
#
# The Req class is used for checking conditional and unconditional requirements. The other classes are the
# specific requirements that can be made - e.g. that an argument is given/not given, or that an argument takes on
# a specific value or range of values.
import numpy as np


class NotGiven():
    """
    Requirement that an argument have value None
    """
    def __init__(self, arg_name):
        if type(arg_name) != list:
            arg_name = [arg_name]
        self.arg_name = arg_name

    def is_true(self, args):
        """
        Check that this requirement is true

        :param args: The args namespace which needs to be validated
        :return:
        """
        return not np.any([getattr(args, a) is not None and getattr(args, a) != 'ignore' for a in self.arg_name])

    def message(self, args):
        """
        Message used for creating error message when this is apart of a requirement that
        has not been met

        :param args: The args namespace which needs to be validated
        :return:
        """
        issues = np.array(self.arg_name)[[hasattr(args, a) for a in self.arg_name]][0]
        return f'specifying {issues}'


class Given():
    """
    Requirement that an argument have a value other than None or 'ignore'
    """
    def __init__(self, arg_name):
        if type(arg_name) != list:
            arg_name = [arg_name]
        self.arg_name = arg_name

    def is_true(self, args):
        """
        Check that this requirement is true

        :param args: The args namespace which needs to be validated
        :return:
        """
        return np.all([getattr(args, a) is not None and getattr(args, a) != 'ignore' for a in self.arg_name])

    def message(self, args):
        """
        Message used for creating error message when this is apart of a requirement that
        has not been met

        :param args: The args namespace which needs to be validated
        :return:
        """
        issues = np.array(self.arg_name)[[not hasattr(args, a) or getattr(args, a) == 'ignore' for a in self.arg_name]]
        return f'not specifying {issues}'


class Values():
    """
    Reqiurement that an argument be an element of a list
    """
    def __init__(self, arg_name, values):
        self.arg_name = arg_name
        assert type(values) == list
        self.values = values

    def is_true(self, args):
        """
        Check that this requirement is true

        :param args: The args namespace which needs to be validated
        :return:
        """
        return getattr(args, self.arg_name) in self.values

    def message(self, args):
        """
        Message used for creating error message when this is apart of a requirement that
        has not been met

        :param args: The args namespace which needs to be validated
        :return:
        """
        return f'{self.arg_name} value of {getattr(args, self.arg_name)}'


class ValueRange():
    """
    Reguire that an argument have value in a specific range
    """
    def __init__(self, arg_name, value_range):
        self.arg_name =arg_name
        self.value_range = value_range
        assert len(value_range) == 2

    def is_true(self, args):
        """
        Check that this requirement is true

        :param args: The args namespace which needs to be validated
        :return:
        """
        val = getattr(args, self.arg_name)
        lower_lim = self.value_range[0]
        upper_lim = self.value_range[1]
        if lower_lim is not None and val < lower_lim:
            return False
        elif upper_lim is not None and val > upper_lim:
            return False
        else:
            return True

    def message(self, args):
        """
        Message used for creating error message when this is apart of a requirement that
        has not been met

        :param args: The args namespace which needs to be validated
        :return:
        """
        return f'{self.arg_name} value outside range of {self.value_range}'


class Req():
    """
    Class for checking conditional or unconditional requirements.

    If two requirements are specified, the first one only needs to be true if the second one is

    If only one requirement is specified, it must always be true

    """
    def __init__(self, b, a=None):
        assert type(b) in [ValueRange, Given, NotGiven, Values]
        self.b = b
        if type(a) != list:
            a = [a]
        self.a = a

    def validate(self, args):
        """
        Check that the requirement has been met

        :param args: The args namespace which needs to be validated
        :return:
        """
        self.args = args
        a_is_true = np.all([a is None or a.is_true(args) for a in self.a])
        if a_is_true and not self.b.is_true(args):
                raise ValueError(self)
        else:
            pass

    def __str__(self):
        """
        Convert to string. Used for error messages.

        :return:
        """
        if self.a[0] is not None:
            a_message = [a.message(self.args) for a in self.a]
            return f'{a_message} cannot be combined with {self.b.message(self.args)}'
        else:
            return f'Error: {self.b.message(self.args)}'