# %% Import
import sys
import os
import numpy as np
# Add dirctory with model moduels to path
sys.path.insert(0, os.path.join(os.getcwd(),'..'))
from CIBUSmod.utils.retriever import ParameterRetriever

# %% Initialise retriever
test = ParameterRetriever(
    os.path.join('..','tests','retriever_test.xlsx')
)
print(test.data)

# %% Get default value without filter when others are available
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

# %% Get values loaded from csv
test.clear()
assert test.get('five', A='a1') == 5.1

test.clear()
assert test.get('five', A='a1', E='e1') == 5.2

test.clear()
assert test.get('five', E='e1') == 5.3

test.clear()
assert np.isnan(test.get('five'))

# %%
