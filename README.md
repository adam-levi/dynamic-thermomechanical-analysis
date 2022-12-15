# dynamic-thermomechanical-analysis
Automates the extraction and analysis of DMA data for several experiment types: creep, strain sweep, multifrequency temperature sweep, and stress relaxation.
DMA_math.py contains various mathamathical functions for data manipulation to determine values of interest.
DMA_extract.py contains functions for removing meta data that is automatically added to raw data files from the TA Q800 DMA instrument.
analysisDMA.py is a class containing the attributes associated with generating values of interest for each experiment type and a function to generate a pandas dataframe from those values. Each experiment type has it's own subclass, containing a specific function for data analysis and data plotting, respectively
dmaAnalysisNotebook.ipynb is a Jupyter notebook demonstrating the use of each of these functions for each experiment type.
