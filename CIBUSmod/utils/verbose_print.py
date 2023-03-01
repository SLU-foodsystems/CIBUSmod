import time

# Function for printing progress messages with time-stamps if verbose==True
def verbose_init(verbose, id_str='', head_len=62):
    if verbose:
        tic = time.time()
        if id_str != '':
            id_str = f'[{id_str}]'
        def verbose_print(*args, type='prog', sep=' ',end='\n'):
                        if type=='head':
                            heading = sep.join(args)
                            stars = '*' * int((head_len - len(heading)) / 2 - 2)
                            return print('\n',stars,heading,stars,'\n', sep=sep, end=end)
                        elif type=='end':
                            time_stamp = f'[{time.strftime("%H:%M:%S",time.localtime())}]'
                            return print(f'{time_stamp}{id_str}',*args, f'Done! Elapsed time: {(time.time()-tic):.0f} sec',sep=sep,end=end)
                        else:
                            time_stamp = f'[{time.strftime("%H:%M:%S",time.localtime())}]'
                            return print(f'{time_stamp}{id_str}',*args, sep=sep, end=end)
                        
    else:
        # empty function for not printing
        def verbose_print(*args,**kwargs):
            pass
    
    return verbose_print