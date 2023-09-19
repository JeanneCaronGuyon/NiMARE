import json
from glob import glob
import nibabel as nib
import pandas as pd
import numpy as np
from scipy import ndimage
from os.path import basename, join, isfile
from pathlib import Path
from rich import print

src_folder = Path(__file__).parent
output_folder = Path(__file__).parent


def _local_max(data, affine, min_distance: int):
    """Find all local maxima of the array, separated by at least min_distance.
    Adapted from https://stackoverflow.com/a/22631583/2589328
    Parameters
    ----------
    data : array_like
        3D array of with masked values for cluster.
    min_distance : :obj:`int`
        Minimum distance between local maxima in ``data``, in terms of mm.
    Returns
    -------
    ijk : :obj:`numpy.ndarray`
        (n_foci, 3) array of local maxima indices for cluster.
    vals : :obj:`numpy.ndarray`
        (n_foci,) array of values from data at ijk.
    """
    # Initial identification of subpeaks with minimal minimum distance
    data_max = ndimage.filters.maximum_filter(data, 3)
    maxima = data == data_max
    data_min = ndimage.filters.minimum_filter(data, 3)
    diff = (data_max - data_min) > 0
    maxima[diff == 0] = 0

    labeled, n_subpeaks = ndimage.label(maxima)
    ijk = np.array(ndimage.center_of_mass(data, labeled, range(1, n_subpeaks + 1)))
    ijk = np.round(ijk).astype(int)

    vals = np.apply_along_axis(arr=ijk, axis=1, func1d=_get_val, input_arr=data)

    # Sort subpeaks in cluster in descending order of stat value
    order = (-vals).argsort()
    vals = vals[order]
    ijk = ijk[order, :]
    xyz = nib.affines.apply_affine(affine, ijk)  # Convert to xyz in mm

    # Reduce list of subpeaks based on distance
    keep_idx = np.ones(xyz.shape[0]).astype(bool)
    for i in range(xyz.shape[0]):
        for j in range(i + 1, xyz.shape[0]):
            if keep_idx[i] == 1:
                dist = np.linalg.norm(xyz[i, :] - xyz[j, :])
                keep_idx[j] = dist > min_distance
    ijk = ijk[keep_idx, :]
    vals = vals[keep_idx]
    return ijk, vals


def _get_val(row, input_arr):
    """Small function for extracting values from array based on index."""
    i, j, k = row
    return input_arr[i, j, k]


def create_dset_from_nidm(src_folder: Path, output_folder: Path):
    ddict = {}
    folders = sorted(src_folder.glob("**/*.nidm"))

    print(f"Found {len(folders)} folders in src_folder: {src_folder}")

    for folder in folders:
        name = basename(folder)
        ddict[name] = {"contrasts": {}}
        ddict[name]["contrasts"]["1"] = {"coords": {}}
        ddict[name]["contrasts"]["1"]["coords"]["space"] = "MNI"
        ddict[name]["contrasts"]["1"]["images"] = {}

        # beta file
        files = glob(join(folder, "Contrast*.nii.gz"))
        files = [f for f in files if "StandardError" not in basename(f)]
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["beta"] = f

        # se file
        files = glob(join(folder, "ContrastStandardError*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["se"] = f

        # z file
        files = glob(join(folder, "ZStatistic*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["z"] = f
        
        # t file
        files = glob(join(folder, "TStatistic*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["t"] = f
        
        # metadata 
        ddict[name]["contrasts"]["1"]["metadata"] = {}
        
        # sample size
        f = join(folder, "DesignMatrix.csv")
        if isfile(f):
            df = pd.read_csv(f, header=None)
            n = [df.shape[0]]
        else:
            n = None
        ddict[name]["contrasts"]["1"]["metadata"]["sample_sizes"] = n

        # foci
        files = glob(join(folder, "ExcursionSet*.nii.gz"))
        f = sorted(files)[0]
        img = nib.load(f)
        data = np.nan_to_num(img.get_fdata())

        # positive clusters
        binarized = np.copy(data)
        binarized[binarized > 0] = 1
        binarized[binarized < 0] = 0
        binarized = binarized.astype(int)
        labeled = ndimage.label(binarized, np.ones((3, 3, 3)))[0]
        clust_ids = sorted(list(np.unique(labeled)[1:]))
        ijk = np.hstack(
            [np.where(data * (labeled == c) == np.max(data * (labeled == c))) for c in clust_ids]
        )
        ijk = ijk.T
        xyz = nib.affines.apply_affine(img.affine, ijk)

        ddict[name]["contrasts"]["1"]["coords"]["x"] = list(xyz[:, 0])
        ddict[name]["contrasts"]["1"]["coords"]["y"] = list(xyz[:, 1])
        ddict[name]["contrasts"]["1"]["coords"]["z"] = list(xyz[:, 2])

    f1 = output_folder / "dset.json"
    with open(f1, "w") as fo:
        json.dump(ddict, fo, sort_keys=True, indent=4)


def create_dset_from_nidm_with_subpeaks():
    ddict = {}
    folders = sorted(glob("/Users/tsalo/Downloads/nidm-pain-results/pain_*.nidm"))
    for folder in folders:
        name = basename(folder)
        ddict[name] = {"contrasts": {}}
        ddict[name]["contrasts"]["1"] = {"coords": {}}
        ddict[name]["contrasts"]["1"]["coords"]["space"] = "MNI"
        ddict[name]["contrasts"]["1"]["images"] = {"space": "MNI_2mm"}

        # beta file
        files = glob(join(folder, "Contrast*.nii.gz"))
        files = [f for f in files if "StandardError" not in basename(f)]
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["beta"] = f

        # se file
        files = glob(join(folder, "ContrastStandardError*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["se"] = f

        # z file
        files = glob(join(folder, "ZStatistic*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["z"] = f

        # t file

        # z file
        files = glob(join(folder, "TStatistic*.nii.gz"))
        f = sorted(files)[0] if files else None
        ddict[name]["contrasts"]["1"]["images"]["t"] = f

        # sample size
        f = join(folder, "DesignMatrix.csv")
        if isfile(f):
            df = pd.read_csv(f, header=None)
            n = [df.shape[0]]
        else:
            n = None
        ddict[name]["contrasts"]["1"]["sample_sizes"] = n

        # foci
        files = glob(join(folder, "ExcursionSet*.nii.gz"))
        f = sorted(files)[0]
        img = nib.load(f)
        data = np.nan_to_num(img.get_fdata())

        # positive clusters
        binarized = np.copy(data)
        binarized[binarized > 0] = 1
        binarized[binarized < 0] = 0
        binarized = binarized.astype(int)
        labeled = ndimage.label(binarized, np.ones((3, 3, 3)))[0]
        clust_ids = sorted(list(np.unique(labeled)[1:]))

        peak_vals = np.array([np.max(data * (labeled == c)) for c in clust_ids])
        clust_ids = [clust_ids[c] for c in (-peak_vals).argsort()]  # Sort by descending max value

        ijk = []
        for c_val in clust_ids:
            cluster_mask = labeled == c_val
            masked_data = data * cluster_mask

            # Get peaks, subpeaks and associated statistics
            subpeak_ijk, subpeak_vals = _local_max(masked_data, img.affine, min_distance=8)

            # Only report peak and, at most, top 3 subpeaks.
            n_subpeaks = np.min((len(subpeak_vals), 4))
            # n_subpeaks = len(subpeak_vals)
            subpeak_ijk = subpeak_ijk[:n_subpeaks, :]
            ijk.append(subpeak_ijk)
        ijk = np.vstack(ijk)
        xyz = nib.affines.apply_affine(img.affine, ijk)
        ddict[name]["contrasts"]["1"]["coords"]["x"] = list(xyz[:, 0])
        ddict[name]["contrasts"]["1"]["coords"]["y"] = list(xyz[:, 1])
        ddict[name]["contrasts"]["1"]["coords"]["z"] = list(xyz[:, 2])

    f2 = output_folder / "dset_with_subpeaks.json"
    with open(f2, "w") as fo:
        json.dump(ddict, fo, sort_keys=True, indent=4)


def main():
    create_dset_from_nidm(src_folder, output_folder)


if __name__ == "__main__":
    main()
