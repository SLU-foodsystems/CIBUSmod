import os
# if os.path.basename(os.getcwd()) == ('notebooks' or 'CIBUSmod'):
#     root = os.path.join(os.getcwd(), os.pardir)
# else:
#     print('Please make sure to be in the CIBUSmod or noteboook subdirectory before running the notebook')
from CIBUSmod import root

input_path = os.path.abspath(os.path.join(root, 'data/soil/input'))
export_path = os.path.abspath(os.path.join(root, 'data/soil/exported_results'))
temp_path = os.path.abspath(os.path.join(root, 'data/soil/temp_results'))

# print(f'root: {root}')
# print(f'input_path: {input_path}')
# print(f'temp_path: {temp_path}')
# print(f'export_path: {export_path}')
