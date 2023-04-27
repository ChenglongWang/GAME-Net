DockOnSurf
==========
[![Documentation Status](https://readthedocs.org/projects/dockonsurf/badge/?version=latest)](http://dockonsurf.readthedocs.io/?badge=latest)

DockOnSurf is a program to automatically find the most stable geometry for molecules
on surfaces.

old webpage: https://forge.cbp.ens-lyon.fr/redmine/projects/dockonsurf

current repository: https://gitlab.com/lch_interfaces/dockonsurf

Features
--------
* Generate a handful of adsorbate-surface structures by 
  combining:
  * surface sites
  * adsorbate's anchoring points
  * conformers
  * orientations
  * probe dissociation of acidic H
  
* Guess the direction where to place the adsorbate. 
  Useful for nanoparticles or stepped/kinked surfaces.
  
* Sample different orientations efficiently by using internal angles.

* Detect and correct atomic clashes.

* Optimize the geometry of the generated structures using CP2K or VASP.
  
* Submit jobs to your computing center and check if they have finished normally.

* Track progress by logging all events on a log file.

* Customize the execution by changing chemically meaningful  the edition of a simple input file.

Documentation
-------------
https://dockonsurf.readthedocs.io/

Installation:
-------------
Download the ``dockonsurf`` directory and place it somewhere in your computer,
by typing in your terminal:

    git clone https://gitlab.com/lch_interfaces/dockonsurf

In order to be able to execute DockOnSurf by simply typing `dockonsurf.py` You need 
to add the DockOnSurf directory in your `PATH`. Assuming you download it in your `$HOME`
directory, add `$HOME/dockonsurf` to your `PATH` variable by typing:

    PATH="$PATH:$HOME/dockonsurf/"

If you downloaded it elsewhere, replace `$HOME` for the actual path where your DockOnSurf is 
(where you did the `git clone` command).
If you want to permanently add the DockOnSurf directory in your `PATH` add 
``PATH="$PATH:$HOME/dockonsurf/"`` at the end of your `$HOME/.bashrc` file.

DockOnSurf needs the python libraries listed under **Requirements** to be installed 
and available. The easiest way to do this is with the `conda` package and environment 
manager (see https://docs.conda.io/en/latest/). You can alternatively install 
them using pip except from RDKit, which is not available as its core routines are 
written in C.

For the intstallation intself, you can check out either
* examples/dockonsurf.yml
 and then run `conda env create -f dockonsurf.yml`, a solution provided by Pierre GIRARD.
 
* Get inspiration from .gitlab-ci.yml and do your installation accordingly

Requirements:
-------------

* [Python](http://www.python.org/) >= 3.6
* [Matplotlib](https://matplotlib.org) ~= 3.2.1
* [NumPy](http://docs.scipy.org/doc/numpy/reference/) >= 1.16.6
* [RDKit](https://rdkit.org/) ~= 2019.9.3
* [scikit-learn](https://scikit-learn.org/) ~= 0.23.1
* [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html) ~= 0.8.26
* [ASE](https://wiki.fysik.dtu.dk/ase/) ~= 3.19.1
* [NetworkX](https://networkx.org/) >= 2.4
* [python-daemon](https://pypi.org/project/python-daemon/) ~= 2.2.4
* [pymatgen](https://pymatgen.org/) ~= 2020.11.11
* [pycp2k](https://github.com/SINGROUP/pycp2k) ~= 0.2.2

Example
-------
Execute DockOnSurf by typing

    dockonsurf.py -i dockonsurf.inp

where `dockonsurf.inp` is the dockonsurf input file. See a sample [here](https://gitlab.com/lch_interfaces/dockonsurf/examples/dockonsurf.inp).

Testing
-------

(To be done)


Contact/Contribute 
-------

Submit an issue on:
https://gitlab.com/lch_interfaces/dockonsurf

Please send us bug-reports, patches, code, ideas and questions.


License
-

DockOnSurf is licensed under the MIT license.
