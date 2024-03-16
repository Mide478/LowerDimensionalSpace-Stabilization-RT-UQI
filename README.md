# LowerDimensionalSpace-Stabilization-RT-UQI


<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-summary"> ➤ Project Summary</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#acknowledgements"> ➤ Acknowledgements</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT SUMMARY -->
<h2 id="project-sumary"> :pencil: Project Summary</h2>

<p align="justify"> 
 
* Current nonlinear manifold dimensionality reduction (NDR) methods have various assumptions about the data, model, subsurface, geological, and engineering settings yielding non-unique solutions and tentatively increased uncertainty.

* We propose an innovative method that stabilizes data representations in lower-dimensional space (LDS) applicable to any manifold dimensionality method using rigid transformations. This aims to create a single unique solution for LDS regardless of data perturbations, starting seed iteration, susceptibility to Euclidean transformations, and tendency of NDR’s to yield non-unique solutions.

* Our method visualizes the uncertainty space for samples in subsurface datasets, which is helpful for model updating and inferential analysis for the inclusion of out-of-sample-points (OOSP) without model recalculation.

* Workflow demonstration on 3 experimental setups via MDS on synthetic and real subsurface datasets using the Euclidean and Manhattan distance metrics over for a large number of model and sample realizations.

</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following open-source packages are mainly used in this project:
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn

Please install other required packages detailed in the `requirements.txt` file and include custom-made `RigidTransformation_UQI_OOSP.py` containing functions in active working directory

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    Scripts
    .
    ├── RigidTransformation_UQI_OOSP.py
    ├── main.py

    Workflows
    .
    ├── Notebook 1: OOSP with Synthetic Data.ipynb
    ├── Notebook 2: OOSP with Real Data.ipynb
    ├── Notebook 3: Result Analysis.ipynb

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The dataset used for this demonstration is publicly available in <a href="[https://github.com/GeostatsGuy](https://github.com/GeostatsGuy/GeoDataSets/blob/master/unconv_MV_v4.csv)"> GeoDataSets: Synthetic Subsurface Data Repository as `unconv_MV_v4.csv` </a> 
  
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ACKNOWLEDGEMENTS -->
<h2 id="acknowledgements"> :scroll: Acknowledgements</h2>
<p align="justify"> 
This work is supported by the Digital Reservoir Characterization Technology (DIRECT) Industry Affiliate Program at the University of Texas at Austin, and many thanks to Equinor for providing real dataset used to test the workflow.
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>  
  👩‍🎓: <b>Ademide O. Mabadeje</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>ademidemabadeje@utexas.edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/Mide478">@Mide478</a> <br>
  
  👨‍🏫: <b>Michael J. Pyrcz</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mpyrcz@austin.utexas.edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/GeostatsGuy">@GeostatsGuy</a> <br>
</p>
<br>
