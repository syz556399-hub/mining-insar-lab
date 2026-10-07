# Scientific foundations and citation boundaries

1. Tan, X.; Song, B.; Bo, H.; Li, Y.; Wang, M.; Lu, G. (2020). *Extraction of Irregularly Shaped Coal Mining Area Induced Ground Subsidence Prediction Based on Probability Integral Method*. Applied Sciences, 10(18), 6623. [doi:10.3390/app10186623](https://doi.org/10.3390/app10186623).
   Use as background for the probability-integral approach. Our implementation uses rectangular analytical integrals and segmented activation; it does not reproduce this paper's Delaunay/Monte Carlo algorithm or validation results.
2. ESA (2007). *InSAR Principles: Guidelines for SAR Interferometry Processing and Interpretation*, TM-19. [Official publication](https://www.esa.int/About_Us/ESA_Publications/InSAR_Principles_Guidelines_for_SAR_Interferometry_Processing_and_Interpretation_br_ESA_TM-19).
   General radar phase, baseline, interferometric geometry and decorrelation background. This project does not implement ESA's full processing chain.
3. Alaska Satellite Facility. *Sentinel-1 InSAR Product Guide*. [Official guide](https://hyp3-docs.asf.alaska.edu/guides/insar_product_guide/), accessed 2026-10-07.
   Supports the distinction between differential products, topographic phase removal, DEM errors and data-quality limitations; it is not a claim that our synthetic terrain reproduces an operational product.

The empirical extraction response, terrain generation, water geometry and coherence attenuation in this project are modeling choices. They are not presented as newly discovered physical laws or formulas validated by these references. Code was developed with AI assistance and checked numerically; scientific conclusions still need external data and domain review.

This project was developed after exploratory comparison with Wu-Patrick/InterferogramSimulator. The current source contains independently written modules rather than imports from that implementation. If a study compares with, uses, or discusses its method/results, cite the relevant original paper. Independent code does not remove the need to cite scientific ideas actually used in a paper. We make no blanket determination about a future manuscript's citation requirements.

Cite this software using CITATION.cff plus the actual repository URL, release/tag and commit used. There is no assigned DOI or associated validation paper at present. Maintainers may replace the collective author entry with verified contributor names before publication.
