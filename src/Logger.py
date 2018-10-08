"""Multi-state logger.

States:
  0 - Muted.
  1 - Only errors.
  2 - Errors and Warnings.
  3 - Errors, Warnings and Info.
  4 - Errors, Warnings, Info and Debug.
"""

from inspect import getframeinfo, stack

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'

class Status:
  ERROR = 1
  WARNING = 2
  INFO = 3
  DEBUG = 4

_status = Status.DEBUG
_add_caller_info = True

def caller_info():
  if _add_caller_info:
    caller = getframeinfo(stack()[2][0])
    return  ' %s:%d:\t' % (caller.filename.split('/')[-1], caller.lineno)
  else:
    return ''

def error(*args):
  if _status >= Status.ERROR:
    print(bcolors.FAIL, 'ERROR:\t%s' % caller_info(), *args, bcolors.ENDC)

def warning(*args):
  if _status >= Status.WARNING:
    print(bcolors.WARNING, 'WARNING:\t%s' % caller_info(), *args, bcolors.ENDC)

def info(*args):
  if _status >= Status.INFO:
    print(bcolors.OKGREEN, 'INFO:\t', *args, bcolors.ENDC)

def debug(*args):
  if _status >= Status.DEBUG:
    print(bcolors.OKBLUE, 'DEBUG:\t%s' % caller_info(), *args, bcolors.ENDC)

def set_status(new_status):
  global _status
  _status = new_status

def set_caller_info(new_value):
  global _add_caller_info
  _add_caller_info = new_value
