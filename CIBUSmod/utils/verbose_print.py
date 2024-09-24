import time

# Function for printing progress messages with time-stamps if verbose==True
def verbose_init(verbose, id_str=''):
    if verbose:
        tic = time.time()
        sub_tic = [None]
        if id_str != '':
            id_str = f'[{id_str}]'
        def verbose_print(*args, type='prog'):
            
            if type != 'msg':
                if sub_tic[0] is not None:
                    sub_timing = f'{(time.time()-sub_tic[0]):.1f}s\n'
                    sub_tic[0] = time.time()
                else:
                    sub_timing = ''
                    sub_tic[0] = time.time()

            if type == 'end':
                time_stamp = f'[{time.strftime("%H:%M:%S",time.localtime())}]'
                return print(f'{sub_timing}{time_stamp}{id_str}',*args, f'Done! Elapsed time: {(time.time()-tic):.0f} sec', sep=' ', end='\n')
            elif type == 'msg':
                return print(*args, sep=' ', end=' ')
            else:
                time_stamp = f'[{time.strftime("%H:%M:%S",time.localtime())}]'
                return print(f'{sub_timing}{time_stamp}{id_str}', *args, sep=' ', end=' ')
                        
    else:
        # empty function for not printing
        def verbose_print(*args,**kwargs):
            return None
    
    return verbose_print