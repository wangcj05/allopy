Getting Started
===============

Python Support
--------------

Only Python 3.6 and above are supported. We recommend using the the Anaconda distribution as it bundles all the required software nicely for you.

Otherwise, you'll have to manage your environment setup (i.e. C-compiler setup and others) yourself if you choose to use pip.

You can download the `Anaconda <https://www.anaconda.com/distribution>`_ or the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution to get started. Miniconda is more bare-bones (smaller) and is thus faster to download and setup.

Installing the Packages
-----------------------

You can install the packages via :code:`conda` or :code:`pypi`. If installing via :code:`conda`, make sure you have added the **conda-forge** channel. The details to do so are listed in the `Configuring Conda`_ section.

.. code-block:: bash

    # conda
    conda install -c danielbok allopy

    # pip
    pip install allopy

Configuring Conda
-----------------

We require packages from both the :code:`conda-forge` and :code:`danielbok` channels. Before anything, open your command prompt and type this command in:

.. code-block:: bash

    conda config --prepend channels conda-forge --append channels danielbok

This command places the **conda-forge** channel to the top of list while the **danielbok** channel will be placed at the bottom. It means that whenever you install packages, :code:`conda` will first look for the package from **conda-forge**. If it can't find the package, it will move down the list to find the package in the other channels. Once it finds the package, it will install it. Otherwise, it will throw an error.

You may get an error message that reads

.. code-block:: bash

    'conda' is not recognized as an internal or external command, operable program or batch file.

In this case, it means that you have not added conda to your path. What you need to do is find the folder you installed the Miniconda or Anaconda package and add them to path.

Assuming you're using a Windows machine and have installed Miniconda to the folder :code:`C:\\Miniconda3\\`, there are 2 ways to add :code:`conda` to your path.

Method 1
~~~~~~~~

1. In the **Start Menu**, search for **edit environment variables for your account**
2. In the top prompt titled **User variables for <NTID>**, search for **PATH**
3. Double click **PATH**
4. In the **Variable Value** section, go to the end and add the following line :code:`;C:\Miniconda3\;C:\Miniconda3\condabin`.
5. Ensure that you have **added** and **not replaced**!!
6. Click **Okay** to everything

Method 2
~~~~~~~~
The second way is to run the following line in your command prompt. However this is not recommended.

.. code-block:: batch

    setx PATH "C:\Miniconda3\;C:\Miniconda3\condabin;%PATH%"

Using Environments
------------------

A tutorial on how to manage your :code:`conda` environment can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

It is best practice to start your project in a new environment.
