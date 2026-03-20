import sys
import time
import threading

_ACTIVE_VERBOSE = None

# Function for printing progress messages with time-stamps if verbose==True
def verbose_init(verbose, id_str='', spinner_delay=1, spinner_interval=0.5):
    global _ACTIVE_VERBOSE

    if _ACTIVE_VERBOSE is not None:
        _ACTIVE_VERBOSE['stop_and_finalize']()
        _ACTIVE_VERBOSE = None

    if not verbose:
        def verbose_print(*args, **kwargs):
            return None
        return verbose_print

    tic = time.time()
    sub_tic = [None]

    if id_str != '':
        id_str = f'[{id_str}]'

    print_lock = threading.Lock()

    class DelayedDotSpinner:
        def __init__(self, delay=0.3, interval=0.5):
            self.delay = delay
            self.interval = interval
            self._thread = None
            self._stop_event = threading.Event()
            self._base_text = ''
            self._visible = False

        def _spin(self):
            start = time.time()

            while not self._stop_event.is_set():
                if time.time() - start >= self.delay:
                    break
                time.sleep(0.01)

            if self._stop_event.is_set():
                return

            self._visible = True
            while not self._stop_event.is_set():
                for n in (1, 2, 3):
                    if self._stop_event.is_set():
                        return
                    with print_lock:
                        sys.stdout.write('\r' + self._base_text + '.' * n + '   ')
                        sys.stdout.flush()
                    time.sleep(self.interval)

        def start(self, base_text):
            self.stop()
            self._base_text = base_text.rstrip('. ')
            self._visible = False
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._spin, daemon=True)
            self._thread.start()

        def stop(self):
            if self._thread is not None:
                self._stop_event.set()
                self._thread.join()
                self._thread = None

        def finalize(self, suffix=''):
            with print_lock:
                line = self._base_text + '...'
                if suffix:
                    line += ' ' + suffix
                sys.stdout.write('\r' + line + '   \n')
                sys.stdout.flush()

    spinner = DelayedDotSpinner(delay=spinner_delay, interval=spinner_interval)
    spinner_active = [False]

    def _progress_elapsed():
        now = time.time()
        if sub_tic[0] is None:
            elapsed = now - tic
        else:
            elapsed = now - sub_tic[0]
        sub_tic[0] = now
        return f'{elapsed:.1f}s'

    def stop_and_finalize():
        if spinner_active[0]:
            spinner.stop()
            spinner.finalize(_progress_elapsed())
            spinner_active[0] = False

    _ACTIVE_VERBOSE = {
        'stop_and_finalize': stop_and_finalize
    }

    def verbose_print(*args, type='prog'):
        global _ACTIVE_VERBOSE

        if spinner_active[0]:
            stop_and_finalize()

        if type == 'msg':
            with print_lock:
                print(*args, sep=' ', end=' ')
            return

        time_stamp = f'[{time.strftime("%H:%M:%S", time.localtime())}]'
        prefix = f'{time_stamp}{id_str}'

        if type == 'end':
            with print_lock:
                print(
                    prefix,
                    *args,
                    f'Done! Elapsed time: {(time.time() - tic):.0f} sec',
                    sep=' ',
                    end='\n'
                )
            _ACTIVE_VERBOSE = None
            return

        base_text = f'{prefix} ' + ' '.join(str(a) for a in args)
        spinner.start(base_text)
        spinner_active[0] = True

    return verbose_print