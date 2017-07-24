# Add the parent folder into sys.path. Ensure sys.path contains no duplicates.

import os
import sys

def unique_list(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

package_par_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# package_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# print('package_folder: {}'.format(package_folder))
# package_par_folder = os.path.abspath(os.path.join(package_folder, os.pardir))
sys.path.insert(0, package_par_folder)
# sys.path.insert(0, package_folder)
sys.path = unique_list(sys.path)
# print("Added folders to PYTHONPATH : {}, {}".format(package_folder, package_par_folder))
print("Added folders to PYTHONPATH : {}".format(package_par_folder))
# print(sys.path)
