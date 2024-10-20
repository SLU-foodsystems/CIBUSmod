# %% Import
import sys
import os
import numpy as np
from time import time
# Add dirctory with model moduels to path
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
from CIBUSmod.utils.retriever import ParameterRetriever

# %% Set data folders and initialise retriever
ParameterRetriever.set_data_folder(os.path.join('./retriever_test_data'))

test = ParameterRetriever(
    name='retriever_test'
)
print(test.data)

# %% Get relation
assert test.get_rel('A','AG') == \
{'a1': 'ag1', 'a2': 'ag1', 'a3': 'ag2', 'a4': 'ag2'}

# %% Get default value without filter when other values are available
test.clear()
assert test.get('one') == 1.1

# %% Get default value with filters that are not matched
test.clear()
assert test.get('one',A='a2') == 1.1

test.clear()
assert test.get('one',F='a2') == 1.1

# Filer not in data
test.clear()
assert test.get('one',H='h1') == 1.1

# %% Get default value without filter with only one available
test.clear()
assert test.get('two') == 2

# %% Get values loaded from csv files
test.clear()
assert test.get('five', A='a1') == 5.1

test.clear()
assert test.get('five', A='a1', E='e1') == 5.2

test.clear()
assert test.get('five', E='e1') == 5.3

test.clear()
assert test.get('six', A='a1', F='f1') == 6.1

test.clear()
assert test.get('six', A='a1', E='e1', F='f1') == 6.2

test.clear()
assert test.get('six', E='e1', F='f3') == 6.9

# %% Return NaN if no propper match

# OK
test.clear()
assert test.get('three', A='a2',B='b2',C='c2') == 3.1

# OK
test.clear()
assert test.get('three', A='a2',B='b2',D='d2') == 3.2

# No match --> NaN and warns
test.clear()
assert np.isnan(test.get('three', A='a1',B='b2',C='c2'))

# Multiple matches --> NaN and warns
test.clear()
assert np.isnan(test.get('three', A='a2',B='b2',C='c2',D='d2'))

# No filter + no default --> NaN and warns
test.clear()
assert np.isnan(test.get('five'))

# %% Return array with correct length when using filters not in data

test.clear()
sel_len = 10
assert len(test.get('one', X=['x']*sel_len)) == sel_len

# %% Get multiple parameres 
test.clear()
tic = time()

n = 30000
res = np.all(
    test.get(
        'seven',
        A=['a1']*n,
        B=['b1']*n
    ) == np.array([7.1]*n)
)
print(f'30,000*1: {round(time()-tic,3)} sec')
assert res

n = 300000
res = np.all(
    test.get(
        'seven',
        A=['a1']*n,
        B=['b1']*n
    ) == np.array([7.1]*n)
)
print(f'300,000*1: {round(time()-tic,3)} sec')
assert res

n = 10000
res = np.all(
    test.get(
        'seven',
        A=['a1','a2','a3']*n,
        B=['b1','b2','b3']*n
    ) == np.array([7.1,7.2,7.3]*n)
)
print(f'10,000*3: {round(time()-tic,3)} sec')
assert res

n = 100000
res = np.all(
    test.get(
        'seven',
        A=['a1','a2','a3']*n,
        B=['b1','b2','b3']*n
    ) == np.array([7.1,7.2,7.3]*n)
)
print(f'100,000*3: {round(time()-tic,3)} sec')
assert res

# Filer not in data
test.clear()
tic = time()
n = 100000
res = np.all(
    test.get(
        'one',
        H=['h1','h2','h3']*n
    ) == np.array([1.1]*3*n)
)
print(f'Unused filters: {round(time()-tic,3)} sec')
assert res

# %% Update from one scenario
test.update_parameter_values('retriever_test_scn1',10)
print(test.data)

# Relative
test.clear()
assert test.get('one') == 11

# Exact scn match 
test.clear()
assert test.get('one', A='a1', B='b1') == 2.6

# less exact scn match
test.clear()
assert test.get('one', A='a1', B='b1', C='c1') == 2.8

# Default scn match
test.clear()
assert test.get('one', A='a1') == 12

# Absolute
test.clear()
assert test.get('two') == 20

# %% One scenario interpolated values
test.update_parameter_values('retriever_test_scn1',5)
print(test.data)

# Relative
test.clear()
assert test.get('one') == 5.5

# Absolute
test.clear()
assert test.get('two') == 10

# %% Multiple scenarios
test.update_parameter_values(['retriever_test_scn1','retriever_test_scn2'],10)
print(test.data)

# scn1 value retained
test.clear()
assert test.get('one') == 11

# scn1 value replaced by scn2 value
test.clear()
assert test.get('one', A='a1', B='b1', C='c1') == 140

# scn2 value not pressent in scn1
test.clear()
assert test.get('three', A='a2', C='c2') == 31

# %% Scenario with no filter columns
test.update_parameter_values(['retriever_test_scn3'],10)
print(test.data)

test.clear()
assert test.get('one') == 1100

test.clear()
assert test.get('four', D='d3') == 40000

# %% Scenario interpolation

# Keep default values until first defined scenario value
test.update_parameter_values(['retriever_test_scn3'],3)
test.clear()
assert test.get('five', A='a1') == 5.1
test.update_parameter_values(['retriever_test_scn3'],7)
test.clear()
assert test.get('five', A='a1') == 5.1 * (1+9/5*2)

# Keep last defined scenario value forward
test.update_parameter_values(['retriever_test_scn3'],5)
test.clear()
assert test.get('three', A='a2', C='c2') == 31
test.update_parameter_values(['retriever_test_scn3'],7)
test.clear()
assert test.get('three', A='a2', C='c2') == 31

# %% Wrong scenario name
test.update_parameter_values(['retriever_test_scn3','retriever_test_scn4'],10)
print(test.data)

# %% Translated filter columns
test.update_parameter_values()
test.clear()
# If non-aggregated filter column defined take that
assert test.get('eight', A='a1') == 8.0

assert test.get('eight', A='a2') == 8.1
assert np.isnan(test.get('eight', A='a3'))
assert test.get('eight', A='a3', G='g1') == 8.2
assert test.get('eight', A='a3', G='g1') == 8.2
assert np.isnan(test.get('eight', A='a3', G='g4'))

assert test.get('nine',A='a1',B='b1',G='g1') == 9.1
assert test.get('nine',A='a3',B='b1',G='g4') == 9.2

# %% Update parameters with val_is='new'
test.update_parameter_values('retriever_test_scn1',1)

# New filter value existing filter and parameter
test.clear()
assert test.get('one', A='n1') == 10

# New filter level and parameter
test.clear()
assert test.get('new', NEW='n1') == 1

# %% Update parameters with val_is='new' in multiple scenario workbooks
test.update_parameter_values(['retriever_test_scn1', 'retriever_test_scn2'],5)
test.data

test.clear()
assert test.get('one', A='n1') == 500

test.clear()
assert test.get('new', NEW='n1') == 5

# %% Raise ValueError if appending new data results in identical filter columns
try:
    test.update_parameter_values('retriever_test_scn_error_new',1)
except ValueError as e:
    assert "val_is='new'" in e.args[0], 'Wrong ValueError raised'
else:
    assert 1==0, 'No error raised'
# %% END
