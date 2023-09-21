# Post-impact disk finder

This repository contains a demonstration version of the disk finding algorithm used in Timpe et al. (2023). This disk finder is intended for use with smooth-particle hydrodynamics simulations of pairwise collisions between planetary-size objects (i.e., "giant impacts"). The disk finder should only be run on a post-impact snapshot for which sufficient time has passed following the impact, thereby allowing the system to reach relative equilibrium. In Timpe et al. (2023), for example, an initial check is done prior to invoking the disk finder that ensures that the collision outcome was a merger (i.e., only one major post-impact body remains) and the post-impact state is relatively quiescent. Note that while the disk finder is relatively fast for resolutions up to a few hundred thousand particles, it will slow down as the number of particles increases.

### Timpe et al. (2023)

Title: A Systematic Survey of Moon-Forming Giant Impacts I: Non-rotating bodies

Authors: Miles Timpe, Christian Reinhardt, Thomas Meier, Joachim Stadel, Ben Moore

Corresponding author: Miles Timpe <mtimpe@proton.me>

Affiliation: Insitute for Computational Science, University of Zurich, Switzerland

### Repository structure

This repository contains the code for the disk finding algorithm, as well as an example of a post-impact collision snapshot and a Jupyter notebook to run the disk finder on this example.

The Jupyter notebook demonstration is available in the `disk_finder_tutorial.ipynb` file. The astrophysical constants used in the code and notebook can be found in `constants.py` and the supporting functions in `utils.py`. Note that the disk finder requires several standard Python libraries, as well as the `pynbody` library, which can be found here: https://pynbody.github.io/pynbody

### Citing this disk finder

If you use this disk finding algorithm in your work, please cite the associated publication:

Timpe, M., Reinhardt, C., Meier, T., Stadel, J., & Moore, B.  "A Systematic Survey of Moon-Forming Giant Impacts I: Non-rotating bodies", 2023, Astrophysical Journal.
