import colorama


colorama.init()
Fore = colorama.Fore


class Label:
    """A class to create colored labels are printed when called.
    
    Colors should be ANSI color codes. Default arguments for the label
    can be set when the label is instantiated (and are stored by the
    same name as attributes). When called, all attributes can be
    overwritten as keyword-only arguments, except for `message`, which
    is positional.
    
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
            to '\n'
    
    Examples:
        >>> from colorama import Fore
        >>> class Labels:
        ...    error = Label('Error', Fore.LIGHTRED_EX)
        ...    success = Label('Success', Fore.LIGHTGREEN_EX)
        ...
        >>> Labels.error('error message with red label')
        [Error] error message with red label
        >>> Labels.success('success message with green label')
        [Success] success message with green label
        >>> Labels.error('message', label='Label Overwrite')
        [Label Overwrite] message
    
    """
    
    def __init__(self, label: str, label_color: str=Fore.RESET, message: str=None, message_color: str=Fore.WHITE, *,
                 encasing: tuple[str]=('[', ']'), encasing_color: str=Fore.WHITE, pre: str='', end: str='\n'):
        self.label = label
        self.label_color = label_color
        self.message = message
        self.message_color = message_color
        self.encasing = encasing
        self.encasing_color = encasing_color
        self.pre = pre
        self.end = end
    
    def __repr__(self):
        return (f'Label(label={repr(self.label)}), label_color={repr(self.label_color)}, message={repr(self.message)}, '
                f'message_color={repr(self.message_color)}, encasing={repr(self.encasing)}, '
                f'encasing_color={repr(self.encasing_color)}, pre={repr(self.pre)}, end={repr(self.end)}')
    
    def __len__(self):
        message = '' if self.message is None else self.message
        return sum((1, *map(len, self.label, self.encasing[0], self.encasing[1], message)))
    
    def __call__(self, message: str=None, *, label: str=None, label_color: str=None, message_color: str=None,
                 encasing: tuple[str]=None, encasing_color: str=None, pre: str=None, end: str=None):
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
        print(''.join((pre, encasing_color, encasing[0], label_color, label, Fore.RESET, encasing_color, encasing[1],
                       ' ', message_color, message, Fore.RESET)), end=end)
