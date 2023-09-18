import os
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map

from pathlib import Path

from nimare.dataset import Dataset
from nimare.workflows.ibma import IBMAWorkflow

output_folder = Path(__file__).parent

dset_file = output_folder / "dset.json"

dset = Dataset(str(dset_file),
               mask="/home/remi/github/NiMARE/nimare/resources/templates/MNI152_2x2x2_brainmask.nii.gz")


workflow = IBMAWorkflow()
result = workflow.fit(dset, drop_invalid=False)

###############################################################################
# Plot Results
# -----------------------------------------------------------------------------
# The fit method of the IBMA workflow class returns a :class:`~nimare.results.MetaResult` object,
# where you can access the corrected results of the meta-analysis and diagnostics tables.
#
# Corrected map:
img = result.get_map("z_corr-FDR_method-indep")
plot_stat_map(
    img,
    cut_coords=20,
    display_mode="z",
    threshold=1.65,  # voxel_thresh p < .05, one-tailed
    cmap="RdBu_r",
    vmax=4,
)
plt.show()

###############################################################################
# Clusters table
# ``````````````````````````````````````````````````````````````````````````````
result.tables["z_corr-FDR_method-indep_tab-clust"]

###############################################################################
# Contribution table
# ``````````````````````````````````````````````````````````````````````````````
result.tables["z_corr-FDR_method-indep_diag-Jackknife_tab-counts_tail-positive"]

###############################################################################
# Report
# -----------------------------------------------------------------------------
# Finally, a NiMARE report is generated from the MetaResult.
from nimare.reports.base import run_reports

# root_dir = Path(os.getcwd()).parents[1] / "docs" / "_build"
# Use the previous root to run the documentation locally.
html_dir = output_folder / "workflow"

run_reports(result, html_dir)
