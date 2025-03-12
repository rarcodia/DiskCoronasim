==============================
Disk-corona emission simulator
==============================

This script is a simplified version of the code used in `Arcodia et al. (2019) <https://arxiv.org/abs/1907.10069>`_ for the modelization of the Lx-Luv relation in AGN.

With an input M and Mdot (and also X-ray photon-index and spin), the model will compute monochromatic luminosities for the disk and corona emission, as well as radial profiles for all the quantities of interest (see details below).

Since the model relies on the assumptions of a geometrically-thin and optically-thick accretion disk, it is mostly suitable for radiatively efficient AGN (Mdot between 0.0x-1 Edd.) or X-ray binaries in their disk-dominated soft-state.

Be aware of the model assumptions and caveats (read carefully Sec. 2 and Appendix A of `Arcodia et al., 2019 <https://arxiv.org/abs/1907.10069>`_).

If you find it useful for your research, please cite the paper `Arcodia et al. (2019) <https://arxiv.org/abs/1907.10069>`_ and link this github repository.

Please note that the code is not intended to be a bullet-proof software and the purpose is simply to share a tool that can quickly provide disk and corona luminosities in a self consistent calculation, given a BH ID card (M, Mdot..).

Comments welcome. For any questions, please send me an email at riccardo.arcodia@gmail.com .

==========
How to use
==========

Make sure you have the required python modules installed (see requirements.txt and add the modules to your python3 environment or try: pip3 install -r requirements.txt --user).

Please see a few quick examples on the jupyter notebook "quick_tutorial".

Simply import DiskCoronaSim and Plot, then run the function called DiskCoronaSim.runmodel() with the parameters of interest (see details in DiskCoronaSim.py).

Results will be stored in json files (see details at the end of DiskCoronaSim.py): for instance, the integrated monochromatic disk and corona luminosities are stored there.

Plotting scripts are available (see Plot.py).

===================
Model in a nutshell
===================

The disk-corona model in `Arcodia et al. (2019) <https://arxiv.org/abs/1907.10069>`_ is based on the prescriptions put forward by `Merloni (2003) <https://ui.adsabs.harvard.edu/abs/2003MNRAS.341.1051M/abstract>`_, in which the standard conservation equations of a geometrically-thin and optically-thick accretion disk (`SS73 <https://ui.adsabs.harvard.edu/abs/1973A%26A....24..337S/abstract>`_) are self-consistently coupled with the X-ray corona, indicated as the fraction f of accretion power that is dissipated away from the cold disk.

The accretion power can be expressed in terms of the stress tensor (i.e. the pressure exerted by each annulus on the vertical area separating the annuli), which we want to express in terms of quantities that we are more familiar with, e.g. gas and/or radiation pressure. In this script we adopt by default the prescription of the magnetic stress being proportional to the geometric mean of gas and total (gas+rad) pressure (see `Merloni,2003 <https://ui.adsabs.harvard.edu/abs/2003MNRAS.341.1051M/abstract>`_).

The system is solvable given a BH ID card (M, Mdot, spin..), with a solution of f as function of the distance from the BH, that represents the energetic interplay between disk and corona.

Along with f, all quantities of interest are computed as a function of the distance from the BH: pressure, density, temperature, surface temperature, monochromatic luminosities (for both disk and corona), opacity, scale heigth of the disk..

For details read `Arcodia et al. (2019) <https://arxiv.org/abs/1907.10069>`_ and `Merloni (2003) <https://ui.adsabs.harvard.edu/abs/2003MNRAS.341.1051M/abstract>`_. In particular, Section 2 and Appendix A in `Arcodia et al. (2019) <https://arxiv.org/abs/1907.10069>`_.
