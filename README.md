# Setup environment

1. Create a Conda environment which uses Python 3.9

   ```bash
   conda create -n ENVIRONMENT_NAME python=3.9 -y
   ```

2. Activate the newly created environment.

   ```bash
   conda activate ENVIRONMENT_NAME
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

# Update requirements

**DO THIS ONLY** every time a new library or dependency is added to the Conda environment in order to keep a track of them and ensure
it also works for future usages of this repository.

```bash
pip list --format=freeze > requirements.txt
```

# Guide

If you are starting to use this scripts, it is highly recommended that you follow the scripts in the order specified below:

## Dataset Slicer

A script to limit the number of classes from a JSON file. It groups and sorts every MS-ASL video entry. It returns an output file containing MAX_CLASSES with each video entry for each sign.

**NOTE:** Edit only uppercase constants values under the configuration section at the top of the script.
