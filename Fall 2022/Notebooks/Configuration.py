import papermill as pm

# Define the input and output notebooks
input_notebook = "/Users/ademidemabadeje/Documents/UT/Research/PyCharm/LD_Stabilization/Fall 2022/Notebooks/Trial with autoresampling and class v1.ipynb"


output_notebook = 'N=100.ipynb'

# Define the parameters to be passed into the notebook
parameters = {
    'N': 100

}

# Execute the notebook and pass in the parameters
pm.execute_notebook(
    input_notebook,
    output_notebook,
    parameters=parameters
)
