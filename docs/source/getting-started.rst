Getting Started
===============

Python Support
--------------

Only Python 3.6 and above are supported. We recommend using the the Anaconda distribution as it bundles all the required software nicely for you.

Otherwise, you'll have to manage your environment setup (i.e. C-compiler setup and others) yourself if you choose to use pip.

You can download the `Anaconda <https://www.anaconda.com/distribution>`_ or the `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ distribution to get started. Miniconda is more bare-bones (smaller) and is thus faster to download and setup.

Configuring Conda
-----------------

Before anything, open your command prompt and type this command in:

.. code-block:: batch

    conda config --prepend channels conda-forge

You may get an error message that reads

.. code-block:: batch

    'conda' is not recognized as an internal or external command, operable program or batch file.

In this case, it means that you have not added conda to your path. What you need to do is find the folder you installed the Miniconda or Anaconda package and add them to path. T

Assuming you installed Miniconda to the folder :code:`C:\\Miniconda3\\`, there are 2 ways to add :code:`conda` to your path.

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


Installing Allopy
-----------------

Install :code:`Allopy` from conda using the following commands:

.. code:: batch

    conda install -c danielbok muarch allopy


Using Environments
------------------

A tutorial on how to manage your :code:`conda` environment can be found `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_.

It is best practice to start your project in a new environment.
