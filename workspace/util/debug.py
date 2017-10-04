import inspect
#
# Color terminal (https://stackoverflow.com/questions/287871/print-in-terminal-with-colors-using-python).
class colours:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
#
# Error information.
def Error():
    callerframerecord = inspect.stack()[2]
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    return '%s: __%s__:%d'%(info.filename, info.function, info.lineno)
#
# Print something in color.
def printColor(str, type):
    print(type + str + colours.ENDC)
#
# Print error information.
def PrintFrame():
    printColor(Error(), colours.WARNING)
#
# Print an error.
def printError(errstr):
    msg = 'ERR: %s:  %s'%(Error(), errstr)
    printColor(msg, colours.FAIL)
#
# Print a warning.
def printWarn(warnstr):
    msg = 'WARN: %s:  %s'%(Error(), warnstr)
    printColor(msg, colours.WARNING)
