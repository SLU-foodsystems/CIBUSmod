import os
# if os.path.basename(os.getcwd()) == ('notebooks' or 'CIBUSmod'):
#     root = os.path.join(os.getcwd(), os.pardir)
# else:
#     print('Please make sure to be in the CIBUSmod or noteboook subdirectory before running the notebook')
from CIBUSmod import root

soil_input_path = os.path.abspath(os.path.join(root, 'data/soil/input'))
soil_export_path = os.path.abspath(os.path.join(root, 'data/soil/exported_results'))
soil_temp_path = os.path.abspath(os.path.join(root, 'data/soil/temp_results'))

# print(f'root: {root}')
# print(f'soil_input_path: {soil_input_path}')
# print(f'soil_temp_path: {soil_temp_path}')
# print(f'soil_export_path: {soil_export_path}')
