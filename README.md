# BEAMSETI Catalog Tools

## Overview

beamseti is a Python package providing functions for analyzing and comparing empirical and synthetic astronomical catalogs in the context of the Search for Extraterrestrial Intelligence (SETI). It incorporates pipelines and plotting utilities that process and visualize data from surveys like Gaia, NED, SynthPop, and others, enabling complementary statistical and empirical constraints on radio SETI survey sensitivity and transmitter rate estimates.

Key functionalities include:

- Data processing pipelines (process_gaia, process_synthpop, process_ned, process_uno) that handle multi-band, multi-approach SETI datasets, treating distances, clustering, and telescope receiving bands.
- Visualization tools (plot_logTR_vs_logEIRPmin, plot_density_colored_cmd_catalog_compare) that create color-magnitude diagrams (CMDs) and comparative plots with independent density color scalings and clear overlays.
- Support for flexible user specifications of bands and catalogs with thorough normalization and color mapping ensuring meaningful comparison.

This package supports easy reproducibility and extensible analysis for SETI researchers.

## Table of Contents

- [Technologies and Requirements](##technologies-and-requirements)  
- [Installation Instructions](##installation-instructions)  
- [Usage Examples](##usage-examples)  
- [Data and File Overview](#data-and-file-overview)  
- [Results and Evaluation](#results-and-evaluation)  
- [Contribution Guidelines](#contribution-guidelines)  
- [Contact and Support](#contact-and-support)  
- [Acknowledgments and References](#acknowledgments-and-references)  
- [Roadmap](#roadmap)  

## Technologies and Requirements

- **Programming Language:** Python 3.8+  
- **Core Libraries:**  
  - `numpy` - Numerical operations  
  - `pandas` - Data manipulation  
  - `matplotlib` - Plotting and visualization  
  - `scipy` - Statistical computations  
  - `astropy` - Astronomy utilities  
  - `astroquery` - Astronomical data querying  
  - `s2sphere` - Spherical geometry for accurate sky coverage calculations  
  - `synthpop` - Synthetic Galactic stellar population synthesis (external package by Klüter & Huston et al. 2025)


## Installation Instructions

1. Clone the repository:
   git clone https://github.com/matti656/beamseti.git

   cd beamseti

2. (Optional) Create and activate a virtual environment:
   python -m venv my_venv

   source my_venv/bin/activate  # Mac/Linux

   my_venv\Scripts\activate     # Windows

3. Install the package with dependencies:
   pip install -e .

   This will install the BEAMSETI package and all required dependencies.

4. Additional setup (if needed):

   - To install or update SynthPop separately:
    Clone Macy's SynthPop repository:

   git clone https://github.com/synthpop-galaxy/synthpop.git

   cd synthpop

   pip install -e .

   pip install ebfpy

## Usage Examples

**Refer to function docstrings in notebooks folder for detailed input parameters and expected output formats.**

Import the key functions from the package:

from beamseti import process_gaia, process_synthpop, process_ned, process_uno

from beamseti.plotting import plot_logTR_vs_logEIRPmin, plot_density_colored_cmd_catalog_compare

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import itertools

# Example: Process Gaia data
gaia_df = pd.read_csv('path/to/field_data.csv')

result_gaia = process_gaia(gaia_df,... split_by_band=True)

# Example: Compare CMDs overlay of Gaia and SynthPop across bands
catalogs = {
    'Gaia': {'GBT-L': gaia_df_l, 'GBT-S': gaia_df_s},
    'SynthPop': {'GBT-L': synthpop_df_l}
}

fig, ax = plot_density_colored_cmd_catalog_compare(catalogs, overlay=True)


# Data and File Overview

- Input Data:
  - Gaia catalogs with columns like `bp_rp` and `abs_g_photogeo`.  
  - SynthPop synthetic catalogs with Gaia-like photometric columns and distances.  
  - NED extragalactic catalog data including redshifts and group membership.  
  - Uno survey data with stellar mass functions and field overlaps.

- File Organization:**  
  - beamseti/ — Python package modules with core functions and plotting utilities.  
  - notebooks/ —   Docstrings for each pipeline.
  - setup.py and requirements.txt — For package installation and dependency management.  
  - tutorial.ipynb - Example Jupyter notebooks demonstrating data processing and visualization workflows.

- Data Preprocessing:
  - Dataframes expected to be cleaned for missing values and correct ra/dec/fwhm_arcmin/fmin/nu_rel/field name/receiving_band columns before processing.  
  - Receiving bands and telescope field identifiers included as columns for multi-band processing.

# Results and Evaluation

- The pipelines produce CMDs with independently normalized density color scales, enabling comparison of different 
    receiving bands  and survey approaches.  
- Transmitter rate versus minimum detectable EIRP plots (plot_logTR_vs_logEIRPmin) allow overlaid or side-by-side 
    comparison and estimation of SETI survey sensitivity and prevalence constraints.  
- Clustering and redshift-independent distance estimation improve the physical realism of population limits.  
- Detailed uncertainty quantification for key parameters is incorporated based on Gaia query crossmatched with Bailer-Jones et al. (2021) distance estimates and SynthPop Poisson statistics.

Example visualizations are provided in the notebooks/ folder to illustrate these outputs.

# Contribution Guidelines

Contributions are welcome! To contribute:

1. Fork the repository.  
2. Create a feature branch (git checkout -b feature-name).  
3. Commit your changes (git commit -am 'Add feature').  
4. Push to the branch (git push origin feature-name).  
5. Open a Pull Request.

Please include clear documentation and tests with new features or bug fixes. Report issues via the GitHub issue tracker.

# Contact and Support

Maintained by Matti Weiss, Berkeley SETI Research Center Intern at Breakthrough Listen
Email: weissm@bxscience.edu  
GitHub: https://github.com/matti656/SETI

For support and questions, please open an issue on GitHub or contact via email.

# Acknowledgments and References

- SynthPop framework by Klüter & Huston et al. (2025), GitHub: https://github.com/synthpop-galaxy/synthpop 
- Gaia EDR3 and NED astronomical catalogs  
- Wlodarczyk-Sroka, B. S., Garrett, M. A., Siemion, A. P. V. (2020), "Extending the Breakthrough Listen nearby star survey 
    to other stellar objects in the field," MNRAS, 498, 5720. https://doi.org/10.1093/mnras/staa2672
    For methodology on empirical galactic SETI beam content analysis
- Bailer-Jones, C.A.L., et al. (2021), "Estimating Distances from Parallaxes. V. Geometric and Photogeometric Distances to 
    1.47 Billion Stars in Gaia Early Data Release 3," AJ, 161, 147. https://iopscience.iop.org/article/10.3847/1538-3881/abd806
    For distance estimates used in Gaia processing and analysis
- Klüter, J., Huston, M. J., Aronica, A., et al. (2025), "SynthPop: A New Framework for Synthetic Milky Way Population 
    Generation," AJ, 169, 317. https://doi.org/10.3847/1538-3881/adcd7a
    For methodology on statistical galactic SETI beam content analysis
- M A Garrett, A P V Siemion, Constraints on extragalactic transmitters via Breakthrough Listen observations 
    of background sources, Monthly Notices of the Royal Astronomical Society, Volume 519, Issue 3, March 2023, 
    Pages 4581–4588, https://doi.org/10.1093/mnras/stac2607
    For methodology on empirical extragalactic SETI beam content analysis, primarily using NED
- Uno, Y., Hashimoto, T., Goto, T., et al. (2023), "Upper limits on transmitter rate of extragalactic civilizations
    placed by Breakthrough Listen observations," MNRAS, 522, 4649. https://doi.org/10.1093/mnras/stad993
    For methodology on statistical extragalactic SETI beam content analysis, primarily using the GSMF provided by:
- I. K. Baldry, S. P. Driver, J. Loveday, E. N. Taylor, L. S. Kelvin, J. Liske, P. Norberg, A. S. G. Robotham, S. Brough,
    A. M. Hopkins, S. P. Bamford, J. A. Peacock, J. Bland-Hawthorn, C. J. Conselice, S. M. Croom, D. H. Jones, H. R. Parkinson, C. C. Popescu, M. Prescott, R. G. Sharp, R. J. Tuffs, Galaxy And Mass Assembly (GAMA): the galaxy stellar mass function at z < 0.06, Monthly Notices of the Royal Astronomical Society, Volume 421, Issue 1, March 2012, Pages 621–634, https://doi.org/10.1111/j.1365-2966.2012.20340.x
- Python packages: NumPy, pandas, Matplotlib, SciPy, Astropy, Astroquery, s2sphere  

# Roadmap

Planned enhancements include:  

- Expanded support for additional SETI survey catalogs and receiver bands.  
- Improve NED clustering deduplication & extend redshift/distance coverage. 
- Refine stellar population assumptions for NED and Uno.
- Incorporate de-extinction for Gaia.
- Apply code to new Breakthrough survey fields and results.
- Improved uncertainty quantification and error propagation in pipelines.  
- Additional interactive visualization tools and web-based dashboards.  
- Integration with Drake equation parameters for intermittent waveform and time domain consideration.  

Thank you for exploring the SETI Catalog Tools. I hope it accelerates your research in the exciting search for extraterrestrial intelligence!

