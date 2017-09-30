import signal
import time
#
# Handle stopping the thread from a signal.
# https://stackoverflow.com/questions/18499497/how-to-process-sigterm-signal-gracefully
class SigHandler():
  exit = False
  #
  # Initialize.
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)
  def exit_gracefully(self,signum, frame):
    self.exit = True
