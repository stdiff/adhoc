# adhoc

- Build Status:
  [![Build Status](https://travis-ci.org/stdiff/adhoc.svg?branch=master)](https://travis-ci.org/stdiff/adhoc) (master)
  [![Build Status](https://travis-ci.org/stdiff/adhoc.svg?branch=dev)](https://travis-ci.org/stdiff/adhoc) (dev)
- Code Coverage:
  [![codecov](https://codecov.io/gh/stdiff/adhoc/branch/master/graph/badge.svg)](https://codecov.io/gh/stdiff/adhoc) (master)
  [![codecov](https://codecov.io/gh/stdiff/adhoc/branch/dev/graph/badge.svg)](https://codecov.io/gh/stdiff/adhoc) (dev)

## Goal of this repository/library 

We often have to do the same thing for each analysis. 

- create a jupyter notebook and import numpy, pandas, matplotlib, etc.
- put *watermark* to log your environment
- count the missing values in the data set  
- check the data types and correct them if they are wrong
- train math model with an usual grid parameter.
- draw the ROC curve of the trained model
- etc.

This module provides generic classes and functions which can be applied 
almost everywhere and documentation about 

- how to prepare your analysis environment.
- hints of useful commands

### Read Notebooks on nbviewer

- [processing](https://nbviewer.jupyter.org/github/stdiff/adhoc/blob/dev/notebooks/usage-processing.ipynb)
- [modeling](https://nbviewer.jupyter.org/github/stdiff/adhoc/blob/dev/notebooks/usage-modeling.ipynb)


## Setup: Python

Supported Python version: `3.7`

If you want to use `virtualenv`, you can create a new environment
by the following command

    > python -m venv your_env
    
In the working directory you can find a directory `your_env` for 
the environment. You can activate the environment by  

    > source your_env/bin/activate

    
### Libraries 
     
you can find a minimal set of libraries for ad hoc analysis.
    
    (your_env) > pip install -r requirements.txt

Probably you need more libraries. After installing them, you should keep
the list of installed libraries by   

    (your_env) > pip freeze > requirements.txt
    

#### jupytext

[jupytext](https://github.com/mwouts/jupytext) generate a Python script
if you create a jupyter notebook and synchronize the pair. The points are

- You can reconstruct your jupyter notebook from the paired Python script.
- While it is difficult to understand a change of a jupyter notebook by 
  reading the raw text, it is easy to understand a change on a Python 
  script. 
  
Namely jupytext makes it easy to manage notebooks on a git repository.


#### watermark

With this library you can put your environment on your notebook briefly.
Write the following two lines in a cell and let it run.

    %load_ext watermark
    %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn


#### Install "adhoc"

    pip install https://github.com/stdiff/adhoc/archive/v0.4.zip
    
Note that this library is not registered in PyPI, therefore the line 

    adhoc==0.4
    
in `requirements.txt` raises an error. To avoid this error you can put 
the following line instead of the above line 

    git+git://github.com/stdiff/adhoc.git@v0.4#egg=adhoc


## Setup: JupyterLab

There are some useful extensions for Jupyter lab.

- If you do not know the path to `jupyter_notebook_config.py`, then 
  `jupyter --paths` command shows the directories where you might find
  the config file.
- If you have not created it yet, execute `jupyter notebook --generate-config`.
- `jupyter labextension list` shows the list of installed jupyter extensions


### Spell checker

[Spellchecker](https://github.com/ijmbarr/jupyterlab_spellchecker) 
works on Markdown cells and highlights misspelled words, but this 
does not correct them. 


### JupyterLab Template

[This extension](https://github.com/timkpaine/jupyterlab_templates)
enables us to use a notebook template very easily. Therefore you 
do not need to type the same import statements.

Note that the extension looks for templates files under the *subdirectories*
of the specified directories. 


### DrawIO

With [DrawIO](https://github.com/QuantStack/jupyterlab-drawio) 
you can draw diagrams easily.

NB. This might not wort because of 
[this issue](https://github.com/jupyterlab/jupyterlab/issues/3506#issuecomment-586510580). 


## useful references 

### pandas

- [visualization](https://pandas.pydata.org/docs/user_guide/visualization.html)


### matplotlib

- [list of cmap strings](https://matplotlib.org/examples/color/colormaps_reference.html)
- [List of named colors](https://matplotlib.org/3.1.0/gallery/color/named_colors.html)


### seaborn 

- [API reference](http://seaborn.pydata.org/api.html)


### sckit-learn

- [scoring parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)

### Altair

- [Official Documentation](https://altair-viz.github.io/)
- [Top-Level Chart Configuration](https://altair-viz.github.io/user_guide/configuration.html)
- [Color Schemes](https://vega.github.io/vega/docs/schemes/)