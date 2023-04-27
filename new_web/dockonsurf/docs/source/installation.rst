Installation
============
Download the ``dockonsurf`` directory and place it somewhere in your computer,
by typing in your terminal: ::

    git clone https://gitlab.com/lch_interfaces/dockonsurf

In order to be able to execute DockOnSurf by simply typing ``dockonsurf.py`` You
need to add the DockOnSurf directory in your ``PATH``. Assuming you download it
in your ``$HOME`` directory, add ``$HOME/dockonsurf`` to your ``PATH`` variable
by typing: ::

    PATH="$PATH:$HOME/dockonsurf/"

If you downloaded it elsewhere, replace ``$HOME`` for the actual path where your
DockOnSurf directory is (where you did the ``git clone`` command).
If you want to permanently add the DockOnSurf directory in your ``PATH`` add 
``PATH="$PATH:$HOME/dockonsurf/"`` at the end of your ``$HOME/.bashrc`` file.

DockOnSurf needs the python libraries listed under **Requirements** to be
installed and available. The easiest way to do this is with the ``conda``
package and environment manager (see https://docs.conda.io/en/latest/). You can
alternatively install them using pip except from RDKit, which is not available
as its core routines are written in C.

Requirements
^^^^^^^^^^^^

* `Python <http://www.python.org/>`_ >= 3.6
* `Matplotlib <https://matplotlib.org>`_ ~= 3.2.1
* `NumPy <http://docs.scipy.org/doc/numpy/reference/>`_ >= 1.16.6
* `RDKit <https://rdkit.org/>`_ ~= 2019.9.3
* `scikit-learn <https://scikit-learn.org/>`_ ~= 0.23.1
* `HDBSCAN <https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html>`_ ~= 0.8.26
* `ASE <https://wiki.fysik.dtu.dk/ase/>`_ ~= 3.19.1
* `NetworkX <https://networkx.org/>`_ >= 2.4
* `python-daemon <https://pypi.org/project/python-daemon/>`_ ~= 2.2.4
* `pymatgen <https://pymatgen.org/>`_ ~= 2020.11.11
* `pycp2k <https://github.com/SINGROUP/pycp2k>`_ ~= 0.2.2

For instance, `this is a hyperlink reference <https://odoo.com>`_.
