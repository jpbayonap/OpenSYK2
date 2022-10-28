import sys
import time
import logging
from qutip.ui import BaseProgressBar


class TextProgressBar(BaseProgressBar):
    """
    A simple text-based progress bar.
    """

    def __init__(self, iterations=0, chunk_size=10):
        super(TextProgressBar, self).start(iterations, chunk_size)

    def start(self, iterations, chunk_size=10):
        super(TextProgressBar, self).start(iterations, chunk_size)

    def update(self, n):
        p = (n / self.N) * 100.0
        if p >= self.p_chunk:
            logging.debug("%4.1f%%." % p +
                  " Run time: %s." % self.time_elapsed() +
                  " Est. time left: %s" % self.time_remaining_est(p))
            # print("%4.1f%%." % p +
            #       " Run time: %s." % self.time_elapsed() +
            #       " Est. time left: %s" % self.time_remaining_est(p))
            sys.stdout.flush()
            self.p_chunk += self.p_chunk_size

    def finished(self):
        self.t_done = time.time()
        print("Total run time: %s" % self.time_elapsed())
