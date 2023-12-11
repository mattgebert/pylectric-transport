import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from scipy.signal import savgol_filter
from pylectric.analysis import mobility
from scipy import optimize as opt



"""Functions for importing
"""

def import_RvB(B, Res, Tht=None, Titles=None):
    """Data preparation for Resistance Data against field.

    Args:
        B (Numpy Array): A 1D list of magnetic field values (Tesla).
        Res (Tuple of Numpy Arrays): Various 1D arrays of resistance (Ohms)
        Tht (Tuple of Numpy Arrays): Various 1D arrays of phase (Degrees). If specified has to match len(R).
        Titles (Tuple of Strings): Various titles of above data. If specified has to mach len(R)

    Returns: 
        Pandas object of data, with appropriate titles.
        Includes calculation
    """
    return ValueError("Not implemented")

def split_bidirectional_sweeps(A):
    """Splits datasweeps that have a bidirectional independent variable.

    Args:
        A (_type_): _description_
    """
    return ValueError("Not implemented")

def remove_duplicate_columns(A, labels):
    """Removes duplicate columns (data), and notes appropriately.

    Args:
        A (numpy.ndarray): Array of 2D data
        labels (list): Column labels

    Returns:
        (np.ndarray, list): Cleaned 2D Data, Cleaned Labels
    """
    
    B = A.copy()
    NCols = B.shape[1]
    # dup_ind = []
    dup_ind = {}
    for i in range(NCols):
        for j in range(1,NCols-i):
            if np.all(A[:,i]==A[:,i+j]):
                # dup_ind.append(i+j)
                if (i+j) in dup_ind:
                    dup_ind[i+j] = dup_ind[i+j].append(i)
                else:
                    dup_ind[i+j]=[i]
    inds = list(set(dup_ind.keys()))
    unique_ind = np.array(inds, dtype=np.dtype(np.int32))
    B = np.delete(B, unique_ind, axis=1)
    new_labels = labels
    if len(unique_ind) != 0:
        print("Removing duplicate data columns:")
        for i in sorted(unique_ind, reverse=True): #reverse deletes preserving index.
            dup_labels = [labels[j] for j in dup_ind[i]]
            print("...\t'" + labels[i] + "', identitcal to ...\n" +\
                "...\t\t(1)...\t" + dup_labels[0])
            if len(dup_labels) > 1:
                for j in range(len(dup_labels)-1):
                    dl = dup_labels[1+j:]
                    print("...\t\t-" + str(j) + ":\t'" + dl + "'")
            del new_labels[i]
    # Search for identical labels but non-identical data reminaing in the dataset.    
    seen = set()
    dupl = []
    for x in new_labels:
        if x not in seen:
            seen.add(x)
        else:
            dupl.append(x)
    print("Warning: Duplicate labels exist with nonmatching data:")
    for l in dupl:
        print("...\t\t" + l)
    return (B, new_labels)

def get_label_duplicate_indexes(labels):
    """Returns indexes where labels match

    Parameters
    ----------
    labels : Array or List
        Array / List of string labels

    Returns
    -------
    Dict
        Dictionary of labels, where values are the duplicate indexes they exist at.
    """
    
    duplicates = {}
    found = set()
    
    for i in range(len(labels)):
        base = labels[i]
        dup = False
        if base not in found:
            for j in range(i, len(labels)):
                if base == labels[j]:
                    if not dup:
                        dup = True
                        found.add(base)
                        duplicates[base] = [i, j]
                    else:
                        duplicates[base].append(j)
    return duplicates

def remove_duplicate_label_columns(A, labels):
    
    """Removes duplicate columns (labels), and notes appropriately.

    Parameters
    ----------
    A : numpy.ndarray
        Array of 2D data
    labels : list
        Column labels

    Returns
    -------
    Tuple(numpy.ndarray, list)
        Cleaned 2D Data, Cleaned Labels
    """
    B = A.copy()
    newlabels = labels.copy()
    duplicates = get_label_duplicate_indexes()
    
    return (B, newlabels)

def arrange_by_label(A, labels, labels_ref):
    """Re-arranges a numpy array based on indexing a list of labels to a reference list of the same labels.
    Assumes all labels are unique, otherwise throws an error. The length of labels and labels ref do not have to match.
    
    Args:
        A (Numpy Array): Data to re-index
        labels (String): Labels to re-index
        labels_ref (String): Reference labels.
    """
    B = A.copy()
    
    assert len(labels) == A.shape[1]
    assert len(labels_ref) <= len(labels)
    
    # Check uniqueness
    if len(labels) != len(set(labels)):
        raise ValueError("Provided labels contain duplicate entries.")
    if len(labels_ref) != len(set(labels_ref)):
        raise ValueError("Provided refernce labels condain duplicate entries.")
    
    ## Check Correspondance
    # for i in set(labels):
    #     if not i in set(labels_ref):
    #         raise ValueError("Provided lables contains strings not in reference labels.")
    
    #Re-arrange A.
    new_indexes = []
    new_labels = []
    extra_indexes = []
    extra_labels = []
    for i in range(len(labels_ref)):
        ref_label = labels_ref[i]
        for j in range(len(labels)):
            label = labels[j]
            if label == ref_label:
                new_indexes.append(j)
                new_labels.append(label)
    for j in range(len(labels)):
        label = labels[j]
        if label not in labels_ref:
            extra_indexes.append(j)
            extra_labels.append(label)
    B = B[:,new_indexes + extra_indexes]
    return B, new_labels + extra_labels



