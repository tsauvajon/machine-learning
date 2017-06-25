# machine-learning

## Getting started

Commands issued to prepare the developping environment on Ubuntu

### Clone the repository
``` shell
clone https://github.com/tsauvajon/machine-learning
```
### Install python 3
``` shell
sudo apt-get install python3
```

### Prepare the virtual environment (to cleanly separate python2 and python3 environments)
``` shell
# install virtualenv
sudo apt-get install virtualenv

# setup the virtual env
virtualenv -p /usr/bin/python3 py3env

# switch to the newly created env
source py3env/bin/activate

# install the required dependencies
pip install scikit-learn numpy scipy
```

### Run the scripts
``` shell
python3 script.py
```
