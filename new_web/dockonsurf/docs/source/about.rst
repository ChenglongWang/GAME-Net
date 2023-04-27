About
=====

DockOnSurf is a program to automatically find the most stable geometry for
molecules on surfaces.

old webpage: https://forge.cbp.ens-lyon.fr/redmine/projects/dockonsurf

current repository: https://gitlab.com/lch_interfaces/dockonsurf/

Features
^^^^^^^^
* Generate a handful of adsorbate-surface structures by combining:

  * surface sites
  * adsorbate's anchoring points
  * conformers
  * orientations
  * probe dissociation of acidic H
  
* Guess the direction where to place the adsorbate.  Useful for nanoparticles or
  stepped/kinked surfaces.
  
* Sample different orientations efficiently by using internal angles.

* Detect and correct atomic clashes.

* Optimize the geometry of the generated structures using CP2K or VASP.
  
* Submit jobs to a computing center and check if they have finished normally.

* Track progress by logging all events on a log file.

* Customize the execution through the edition of a  simple input file.
