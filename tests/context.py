import os
import sys

package_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, package_folder)
print("Added folder to PYTHONPATH : {}".format(package_folder))
print(os.path.abspath(os.path.join(package_folder, os.pardir)))
# print(sys.path)
