import hashlib
import os
import platform as _platform_module
from contextlib import _RedirectStream, suppress
from functools import wraps
from io import StringIO
from threading import Thread
from typing import List, Tuple

from colorama import Fore
with suppress(ImportError):
    import win10toast
    _toaster = win10toast.ToastNotifier()


_platform = _platform_module.system().lower()


class PlatformError(Exception):
    pass


def requires_platform(platform: str):
    """A decorator that raises an error if a function is run on an
    unsupported platform.

    Args:
        platform (str): The platform name. This can be found with
            `platform.system()`. Case is irrelevant.

    Raises:
        PlatformError: If the running platform does not match the one
            dictated in the decorator. This is raised when the decorated
            function is run.

    Examples:
        # if using windows
        >>> @requires_platform('windows')
        ... def f():
        ...     print('Hello, World!')
        >>> f()
        Hello, World!

        >>> @requires_platform('linux')
        ... def f():
        ...     print('Hello, World!')
        >>> f()
        Traceback (most recent call last):
        ...
        PlatformError: this operation requires platform 'linux'

    """
    platform = platform.lower()

    def wrapper(func: object):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if not platform == _platform:
                raise PlatformError(f'this operation requires platform {platform!r}')
            func(*args, **kwargs)
        return wrapped
    return wrapper


def pluralize(word: str, n: int, plural: str='s', append: bool=True) -> str:
    """Pluralize a word.

    Args:
        word (str): The word to pluralize. `str` is called on this.
        n (int): The number that decides if the word should be plural
            or not. If this number is 1, the word will not be
            pluralized, otherwise it will be.
        plural (:obj:`str`, optional): If `append` is True, this string
            will be appended to the word if it should be pluralized. If
            `append` is false, this string will be returned if the word
            should be pluralized. Defaults to 's'.
        append (:obj:`bool`, optional): Whether `plural` should be
            appended to the word (True) or returned in place of the word
            (False). Defaults to True

    Returns:
        str: The plural of `word` if n is not 1. Otherwise return
            `word`. If `append` is True, return `word + plural`,
            otherwise return `plural`.

    Examples:
        >>> pluralize('duck', 2)
        'ducks'
        >>> pluralize('egg', 1)
        'egg'
        >>> pluralize('cactus', 5, 'cacti', False)
        'cacti'

    """
    if n == 1:
        return str(word)
    else:
        if append:
            return str(word) + plural
        else:
            return plural


def run_in_background(func: object):
    """Run `func` in a thread, letting it finish on its own."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        Thread(target=func, args=args, kwargs=kwargs).start()
    return wrapped


@run_in_background
def notify(title: str, message: str=' ', duration: int=5, icon: str=None):
    """Send a windows (only) notification.

    Args:
        title (str): The title of the notification.
        message (:obj:`str`, optional): The message of the
            notification.
        duration (:obj:`int`, optional): The time (in seconds) for the
            notification the show. Defaults to 5.
        icon (:obj:`str`, optional): The path of the icon to use. No
            icon will be displayed if this is None. Defaults to None.
    """
    _toaster.show_toast(title, message, icon, duration)


class Label:
    """A colored label.

    `colorama.init()` needs to be called for colors to work on windows.

    Colors should be selected from `colorama.Fore`. Default arguments
    for the label can be set when the label is instantiated (and are
    stored by the same name as attributes). When called, all attributes
    can be overwritten as keyword-only arguments, except for `message`,
    which is positional.

    Args / Attributes:
        label (str): The label.
        label_color (str): The color of the label, this should be an
            ANSI color code. Defaults to `RESET`.
        message (str): The message. Defaults to None.
        message_color (str): The color of the message, this should be an
            ANSI color code. Defaults to `RESET`.
        encasing (tuple[str]): A tuple of two strings. This is whats
            printed on either side of the label. Defaults to ('[', ']').
        encasing_color (str): The color of the encasing, this should be
            an ANSI color code. Defaults to `RESET`.
        pre (str): The string to be printed before the first encasing.
            Defaults to an empty string.
        end (str): The string to be printed after the message. Defaults
            to '\n'.

    Examples:
        >>> import colorama
        >>> from platform import system
        >>> from colorama import Fore
        >>> if system() == 'Windows':
        ...     colorama.init()
        >>> class Labels:
        ...    error = Label('Error', Fore.LIGHTRED_EX)
        ...    success = Label('Success', Fore.LIGHTGREEN_EX)
        >>> Labels.error('error message with red label')
        [Error] error message with red label
        >>> Labels.success('success message with green label')
        [Success] success message with green label
        >>> Labels.error('message', label='Label Overwrite')
        [Label Overwrite] message
        >>> Labels.success.encasing = ('(', ')')
        >>> Labels.success('success message with green label in parens')
        (Success) success message with green label in parens

    """

    def __init__(self, label: str, label_color=Fore.RESET, message: str=None,
                 message_color=Fore.WHITE, *, encasing: Tuple[str]=('[', ']'),
                 encasing_color=Fore.WHITE, pre: str='', end: str='\n'):
        self.label = label
        self.label_color = label_color
        self.message = message
        self.message_color = message_color
        self.encasing = encasing
        self.encasing_color = encasing_color
        self.pre = pre
        self.end = end

    def __repr__(self):
        return ((f'Label(label={self.label!r}), label_color={self.label_color!r}, '
                 f'message={self.message!r}, message_color={self.message_color!r}, '
                 f'encasing=({self.encasing!r}), encasing_color={self.encasing_color!r}, '
                 f'pre={self.pre!r}, end={self.end!r}'))

    def __len__(self):
        message = '' if self.message is None else self.message
        return sum((1, *map(len, self.label, self.encasing[0], self.encasing[1], message)))

    def __call__(self, message: str=None, *, label: str=None, label_color=None, message_color=None,
                 encasing: tuple=None, encasing_color=None, pre: str=None, end: str=None):
        if message is None:
            if self.message is None:
                message = ''
            else:
                message = self.message
        if label is None:
            label = self.label
        if label_color is None:
            label_color = self.label_color
        if message_color is None:
            message_color = self.message_color
        if encasing is None:
            encasing = self.encasing
        if encasing_color is None:
            encasing_color = self.encasing_color
        if pre is None:
            pre = self.pre
        if end is None:
            end = self.end
        print(''.join((pre, encasing_color, encasing[0], label_color, label, Fore.RESET,
                       encasing_color, encasing[1], ' ', message_color, message, Fore.RESET)),
              end=end)


class AutoInput(_RedirectStream):
    """A context manager to write to stdin with (to automate `input()`).

    Args:
        *args (str): The strings to use as inputs (in the order to be
            used).

    Example:
        >>> with AutoInput('hello') as ai:
        ...     print(input())
        ...     ai.add('eggs', 'spam')
        ...     print(input(), input())
        ...
        hello
        eggs spam
    """

    def __init__(self, *args: str):
        super().__init__(StringIO())
        self._stream = 'stdin'

        self.add(*args)

    def add(self, *args: str):
        location = self._new_target.tell()
        # Go to the end of the stream.
        for _ in self._new_target.readlines():
            pass
        self._new_target.write('\n'.join(args) + '\n')
        self._new_target.seek(location)

    def __enter__(self):
        super().__enter__()
        return self


def auto_input_decorator(*inputs: str):
    """Use `AutoInput` as a decorator.

    Args:
        *inputs (str): The strings to use as inputs (in the order to be
            used).

    Example:
        >>> @auto_input_decorator('hello', 'goodbye')
        ... def func(a):
        ...     print(input())
        ...     print(a)
        ...     print(input())
        >>> func('eggs')
        hello
        eggs
        goodbye

    """
    def wrapper(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with AutoInput(*inputs):
                return func(*args, **kwargs)
        return wrapped
    return wrapper


def hash_file(path: os.PathLike, algorithm: object=hashlib.blake2b, block_size: int=65536) -> bytes:
    """Get the digest of the hash of a file.

    Args:
        path (os.pathlike, str): The path of the file.
        algorithm (object): The hash algorithm object to use. This
            should have an `update` method. Defaults to
            `hashlib.blake2b`.
        block_size (int): The amount of bytes to read into memory at
            once. This should be a multiple of the hash algorithm's
            block size. Defaults to 65536.

    Returns:
        bytes: The digest of the hash.
    """
    with open(path, 'rb') as f:
        hash_ = algorithm()
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            hash_.update(buf)
    return hash_.digest()


def hash_file_hex(path: os.PathLike, algorithm: object=hashlib.blake2b,
                  block_size: int=65536) -> str:
    """Get the hex digest of the hash of a file.

    Args:
        path (os.pathlike, str): The path of the file.
        algorithm (object): The hash algorithm object to use. This
            should have an `update` method. Defaults to
            `hashlib.blake2b`.
        block_size (int): The amount of bytes to read into memory at
            once. This should be a multiple of the hash algorithm's
            block size. Defaults to 65536.

    Returns:
        str: The hex digest of the hash.
    """
    with open(path, 'rb') as f:
        hash_ = algorithm()
        while True:
            buf = f.read(block_size)
            if not buf:
                break
            hash_.update(buf)
    return hash_.hexdigest()


def chunk_list_inplace(lst: list, size: int) -> List[list]:
    """Split a list into chunks (inplace).

    If the list doesn't divide equally, all excess items are appended to
    the end of the output list. To drop the excess items, use
    `chunk_list_inplace_drop_excess`.
    For performance reasons, this function modifies the original list.
    To not modify the original list, use `chunk_list`.

    Args:
        lst(list): The list to chunk.
        size(int): The size of chunks to make.

    Examples:
        >>> chunk_list_inplace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        >>> chunk_list_inplace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        >>> chunk_list_inplace([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
    """
    # Ya know, maybe just rewrite this in C if I care about performance this much. TODO?
    out = []
    while lst:
        out.append(lst[:size])
        del lst[:size]
    return out


def chunk_list_inplace_drop_excess(lst: list, size: int) -> List[list]:
    """Split a list into chunks (inplace).

    If the list doesn't divide equally, all excess items are dropped. To
    keep the excess items, use `chunk_list_inplace`.
    For performance reasons, this function modifies the original list.
    To not modify the original list, use `chunk_list_drop_excess`.

    Args:
        lst(list): The list to chunk.
        size(int): The size of chunks to make.

    Examples:
        >>> chunk_list_inplace_drop_excess(
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        >>> chunk_list_inplace_drop_excess(
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> chunk_list_inplace_drop_excess(
        ...     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    """
    out = chunk_list(lst, size)
    if not len(out[-1]) == size:
        out.pop()
    return out


def chunk_list(lst: list, size: int) -> List[list]:
    """Split a list into chunks.

    If the list doesn't divide equally, all excess items are appended to
    the end of the output list. To drop the excess items, use
    `chunk_list_drop_excess`.
    If the original list is not used after this function is run (and can
    safely be modified), use `chunk_list_inplace` for performace
    reasons.

    Args:
        lst(list): The list to chunk.
        size(int): The size of chunks to make.

    Examples:
        >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]
        >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10]]
    """
    return chunk_list_inplace(list(lst), size)


def chunk_list_drop_excess(lst: list, size: int) -> List[list]:
    """Split a list into chunks.

    If the list doesn't divide equally, all excess items are dropped. To
    keep the excess items, use `chunk_list`.
    If the original list is not used after this function is run (and can
    safely be modified), use `chunk_list_inplace_drop_excess` for
    performace reasons.

    Args:
        lst(list): The list to chunk.
        size(int): The size of chunks to make.

    Examples:
        >>> chunk_list_drop_excess([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 2)
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        >>> chunk_list_drop_excess([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> chunk_list_drop_excess([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 4)
        [[1, 2, 3, 4], [5, 6, 7, 8]]
    """
    return chunk_list_inplace_drop_excess(list(lst), size)


def get_expanded_str(string: str, lst: List[str]):
    """Get the first string of a list that starts with the most
    characters of a given string.

    If the string is empty, return the first item of the list. If the
    list is empty as well, raise a `ValueError`. A `ValueError` will be
    raised if the string is not in the list.

    Args:
        string(str): The string to find caracters in common with.
        lst(list): The list of strings to test against. This list should
            contain strings.

    Rasises:
        ValueError: If no item of the list has any begining characters
            in common with the string.

    Examples:
        >>> get_expanded_str('ro', ['rock', 'paper', 'scissors'])
        'rock'
        >>> get_expanded_str('', ['rock', 'paper', 'scissors'])
        'rock'
        >>> get_expanded_str('rock', ['rock', 'paper', 'scissors'])
        'rock'
        >>> get_expanded_str('egg', ['rock', 'paper', 'scissors'])
        Traceback (most recent call last):
        ...
        ValueError: string 'egg' not in list
    """
    # TODO: clean this up
    if lst:
        if not string:
            return lst[0]
    else:
        raise ValueError(f'string {string!r} not in list')
    scores = {i: 0 for i in lst}
    for i in lst:
        if i == string:
            return i
        score = 0
        with suppress(IndexError):
            for n, char in enumerate(i):
                if not char == string[n]:
                    break
                score += 1
        scores[i] = score
    guess = max(scores.items(), key=lambda i: i[1])
    if len(guess[0]) < len(string) or guess[1] == 0:
        raise ValueError(f'string {string!r} not in list')
    return guess[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
