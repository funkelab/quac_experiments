# Kenyon Cells

The purpose of this experiment is to learn why our classifier consistently mis-classifies Kenyon Cells as dopaminergic instead of cholinergic. 
See discussion in [Ecksteinn et al., 2024](https://www.cell.com/cell/fulltext/S0092-8674(24)00307-6?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0092867424003076%3Fshowall%3Dtrue) for some hypotheses.

To query FlyWire, we use [`fafbseg`](https://fafbseg-py.readthedocs.io/en/latest/source/intro.html). 
Follow the instructions there to get your API key before continuing.


Order of operations: 
1. `get_synapse_locations.py` -> Creates `kenyon_cell_synapses.csv`
2. `run_classification.py` 
    - Creates `kenyon_cell_results` 
    - Adds and fills `location, id, image, prediction` datasets
3. `get_counterfactuals.py`
    - Adds and fills `counterfactual, cf_prediction` groups
    - The datasets within these groups are organized by source and target, e.g. `counterfactual/0/5` dataset is of the same size as `images/0` dataset
4. `run_attribution.py` with a method as an argument (e.g. `integrated_gradients`)
    - Adds `attributions/{method}/`
    - Fills `attributions/{method}/attribution`
5. `evaluate_attribution.py` with a method as an argument
    - Fills `attributions/{method}/mask`
    - Fills `attributions/{method}/hybrid`
