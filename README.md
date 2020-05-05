# BestPractices

Things that you should (and should *not*) do in your Materials Informatics research.

This is a repository containing the relevant Python code and Jupyter notebooks to the publication "Machine Learning for Materials Scientists: An introductory guide towards best practices".

These notebooks are included to illustrate a hypothetical Machine Learning project in materials science created following best practices.
The goal of this project is to predict the heat capacity of materials given a chemical composition and condition (the measurement temperature).

To read the main publication for which these notebooks are made, please see:
```
link coming
```


## Table of Contents
* How to cite
* Installation
* Opening the Jupyter notebooks
* Using the Jupyter notebooks



## How to cite
Please cite the following work if you choose to adopt or adapt the methods mentioned in this editorial:

```
citation coming
```



## Installation
This repositories hosts a series of Jupyter notebooks, which run on the Python programming language.
Please follow the below steps to get started with using these notebooks.


### Clone or download this GitHub repository
Do one of the following:

* [Clone this repository](https://github.com/anthony-wang/BestPractices.git) to a directory of your choice on your computer.
* [Download an archive of this repository](https://github.com/anthony-wang/BestPractices/archive/master.zip) and extract it to a directory of your choice on your computer.


### Install dependencies via Anaconda:
1. Download and install [Anaconda](https://conda.io/docs/index.html).
1. Navigate to the project directory (from above).
1. Open Anaconda prompt in this directory.
1. Run the following command from Anaconda prompt to automatically create an environment from the `conda-env.yml` file:
    - `conda env create --file conda-env.yml`
1. Run the following command from Anaconda prompt to activate the environment:
    - `conda activate bestpractices` (`bestpractices` is the name of the environment)

For more information about creating, managing, and working with Conda environments, please consult the [relevant help page](https://conda.io/docs/user-guide/tasks/manage-environments.html).


### Install dependencies via `pip`:
Open `conda-env.yml` and `pip install` all of the packages listed there.
We recommend that you create a separate Python environment for this project.



## Opening the Jupyter notebooks
We will be using [Jupyter notebooks](https://jupyter.org/) to demonstrate some of the concepts and workflows described in the paper.

Jupyter notebooks give you an interactive development environment, and shows you your code, your code outputs (e.g. calculation results, processed data, visualizations) as well as other rich media (such as text, HTML, images, equations, even LaTeX!) together in a notebook-style environment. Jupyter notebooks are commonly used in the machine learning field.

You should have installed the packages required by Jupyter notebooks already if you followed the steps above to create the `bestpractices` environment.

In that case, you can start Jupyter by following these steps:
1. Navigate to the project directory (from above).
1. Open Anaconda prompt in this directory.
1. Run the following command from Anaconda prompt to start a Jupyter notebook server: `jupyter notebook`
1. Your web browser should open automatically and navigate to the Jupyter notebook webpage that has been created by the Jupyter notebook server. If not, you can look in the Anaconda prompt and find a line that says:
	- The Jupyter Notebook is running at: ```http://localhost:8888/?token=<token>``` (or something similar)
	- Navigate to this address using your favorite web browser to access the Jupyter notebooks
1. In the Jupyter notebook webpage, navigate to the `notebooks` directory and click on a Jupyter notebook to start the notebook
	- We recommend you to start with the first notebook, which provides an overview of the example ML project and the contents of the following notebooks.



## Using the Jupyter notebooks
Jupyter notebooks are composed of several types of "cells", the most import types being code cells ("Code") and text cells ("Markdown").
You can edit code cells by clicking inside the cell and editing the code.
To edit text cells, double click inside the cell and then edit the text. You can use [Markdown-style formatting](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html?highlight=markdown).

You can navigate a Jupyter notebook by using your mouse or your keyboard arrow keys.

Some other handy keyboard shortcuts to know:

| Keyboard shortcut | Description |
| --- | --- |
| `Ctrl + Enter` | Run the contents of a cell |
| `Shift + Enter` | Run the contents of a cell, and then advance to the next cell |
| `Ctrl + S` | Save |

For more information about how to use Jupyter notebooks, you can consult the [official Jupyter Notebook documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html) as well as the wealth of available information online.
