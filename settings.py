"""
plot_efield_profile.py

Author: Victor H. Souza
Date: December 16, 2023

Copyright (c) 2023 Victor H. Souza

Description:
This script reads the paths from the .env file and sets the paths based on the current directory.

Usage:
- Create a .env file in the current working directory
- Ensure that the relative file paths are correctly entered.
- Create a Python environment with the required libraries specified in the environment.yml file

"""

# %%
import os
import dotenv

# %%

dotenv.load_dotenv(dotenv.find_dotenv())

DIR_ROOT = os.getcwd()

DIR_ACOUSTIC = os.path.join(DIR_ROOT, os.getenv('DIR_ACOUSTIC'))

DIR_EFIELD_CURRENT = os.path.join(DIR_ROOT, os.getenv('DIR_EFIELD_CURRENT'))

DIR_MEP = os.path.join(DIR_ROOT, os.getenv('DIR_MEP'))

DIR_MRI = os.path.join(DIR_ROOT, os.getenv('DIR_MRI'))

DIR_SAVE_PLOT = os.path.join(DIR_ROOT, os.getenv('DIR_SAVE_PLOT'))
