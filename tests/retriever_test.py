# %% Import
import sys
import os
import numpy as np
# Add dirctory with model moduels to path
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
from CIBUSmod.utils.retriever import ParameterRetriever

# %% Set data folders and initialise retriever
ParameterRetriever.set_data_folders(
    default = os.path.join('.'),
    scenarios = os.path.join('.')
)

test = ParameterRetriever(
    name='retriever_test'
)
print(test.data)

# %% Get default value without filter when other values are available
test.clear()
assert test.get('one') == 1.1

# %% Get default value with filters that are not matched
test.clear()
assert test.get('one',A='a2') == 1.1

test.clear()
assert test.get('one',F='a2') == 1.1

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

# No match --> NaN
test.clear()
assert np.isnan(test.get('three', A='a1',B='b2',C='c2'))

# Multiple matches --> NaN
test.clear()
assert np.isnan(test.get('three', A='a2',B='b2',C='c2',D='d2'))

# No filter + no default --> NaN
test.clear()
assert np.isnan(test.get('five'))

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

# %%
