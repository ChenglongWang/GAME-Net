.. DockOnSurf documentation master file, created by
   sphinx-quickstart on Thu Feb 11 20:09:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DockOnSurf
==========

DockOnSurf is a program to automatically find the most stable geometry for
molecules on surfaces.

Features
^^^^^^^^
* Generate a handful of adsorbate-surface structures by combining:

  * surface sites
  * adsorbate's anchoring points
  * conformers
  * orientations
  * probe dissociation of acidic H

* Guess the direction where to place the adsorbate. Useful for nanoparticles or
  stepped/kinked surfaces.

* Sample different orientations efficiently by using internal angles.

* Detect and correct atomic clashes.

* Optimize the geometry of the generated structures using CP2K or VASP.

* Submit jobs to a computing center and check if they have finished normally.

* Track progress by logging all events on a log file.

* Customize the execution through the edition of a  simple input file.


.. toctree::
   :maxdepth: 2
   :caption: Contents

   about
   installation
   tutorials
   inp_ref_manual
   tips_and_tricks
   faqs
   release_notes
   contact
   license
   autodoc




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
