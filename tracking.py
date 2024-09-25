import numpy as np
import trackpy as tp
from skimage import img_as_float
from skimage.filters import gaussian, threshold_multiotsu
from skimage.exposure import rescale_intensity
from skimage.morphology import binary_erosion, remove_small_objects, remove_small_holes, disk
from apeer_ometiff_library import io as ome
from pystackreg import StackReg
import pandas as pd
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def get_git_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


def get_pxsize(ome_meta):
    xml = ET.fromstring(ome_meta)
    for image in xml:
        for pixels in image:
            pxsize = pixels.get('PhysicalSizeX')
    return float(pxsize)


def binary(dapi, gauss=1, classes=3, nbins=1000, min_size=15000, area_threshold=5000):
    dapi_g = img_as_float(gaussian(dapi, gauss))
    thresholds = threshold_multiotsu(dapi_g, classes=classes, nbins=nbins)
    mask_general = dapi_g > thresholds[0]
    clean_holes = remove_small_objects(mask_general, min_size=min_size)
    clean = remove_small_holes(clean_holes, area_threshold=area_threshold)
    return clean


def edge(binary):
    selem1 = disk(3)
    inside = binary_erosion(binary, selem1)
    binary[inside] = False
    return binary


def regis(chrom, lamin, fast=False):
    ch_mask = np.zeros_like(chrom, dtype=np.bool_)
    for t in range(chrom.shape[0]):
        ch_mask[t] = binary(lamin[t])
        chrom[t][~ch_mask[t]] = 0

    if fast == False:
        sr = StackReg(StackReg.RIGID_BODY)
        transformations = sr.register_stack(lamin, reference='previous')
        chrom = sr.transform_stack(chrom)
        ch_mask = sr.transform_stack(ch_mask)
        ch_mask = (ch_mask > 0.5)  # kludge
    else:
        pass

    ch_edges = np.copy(ch_mask)
    for k in range(ch_edges.shape[0]):
        ch_edges[k] = edge(ch_edges[k])

    return chrom, ch_mask, ch_edges


def find_dots(stack, minmass=100000, diameter=11, search_range=20, memory=3, threshold=20):
    tp.quiet()
    f = tp.batch(stack, diameter=diameter, minmass=minmass)
    t = tp.link(f, search_range, memory=memory)
    t1 = tp.filter_stubs(t, threshold)
    return t1


def min_dist(df, edges_stack):
    coord_x = np.asarray(list(range(edges_stack[0].shape[1])))
    coord_y = np.asarray(list(range(edges_stack[0].shape[0])))

    xarr = np.zeros(edges_stack[0].shape)
    xarr[:, :] = coord_x
    yarr = np.zeros(edges_stack[0].shape).T
    yarr[:, :] = coord_y

    arr = np.sqrt((xarr - df['x']) ** 2 + (yarr.T - df['y']) ** 2)
    df['min_dist_pxs'] = arr[edges_stack[int(df['frame'])]].min()
    return df


def analyse_data(dots_df, nucl_masks, nucl_edges, pxsize, fname):
    dots_df['y_micron'] = dots_df['y'] * pxsize
    dots_df['x_micron'] = dots_df['x'] * pxsize

    areas = np.sum(nucl_masks, axis=(1, 2)) * pxsize ** 2
    perimeters = np.sum(nucl_edges, axis=(1, 2))
    dots_df['area_micron'] = areas[dots_df['frame']]
    dots_df['perimeter_au'] = perimeters[dots_df['frame']]
    dots_df['perimeter_au_norm'] = dots_df['perimeter_au'] * pxsize ** 2

    data_dist = dots_df.apply(min_dist, axis=1, edges_stack=nucl_edges)
    data_dist['min_dist_micron'] = data_dist['min_dist_pxs'] * pxsize

    data_dist['script_version_git'] = get_git_hash()
    data_dist['serum'] = int(fname.split('_')[1])
    data_dist['date'] = fname.split('_')[0]
    data_dist['file'] = fname[:-4]

    return data_dist



def add_xy_deltas(df):
    c_list = ['file',
              'script_version_git',
              'date',
              'serum',
              'particle',
              'frame',
              'y', 'x',
              'y_micron', 'diff_y_micron',
              'x_micron', 'diff_x_micron',
              'diff_xy_micron',
              'area_micron',
              'perimeter_au',
              'perimeter_au_norm',
              'min_dist_pxs',
              'min_dist_micron',
              ]

    data_a = pd.DataFrame(columns=c_list)
    dots = df['particle'].unique()

    for i in dots:
        data_p = df[df['particle'] == i].copy()
        data_p['diff_y_micron'] = data_p['y_micron'].diff()
        data_p['diff_x_micron'] = data_p['x_micron'].diff()
        data_p['diff_xy_micron'] = ((data_p['diff_x_micron']) ** 2 + (data_p['diff_y_micron']) ** 2) ** .5

        data_a = data_a.append(data_p, ignore_index=True, sort=False)

    data_a = data_a[c_list]

    return data_a


def main():
    inp_path = Path('example_images')
    out_path = Path('tracking_output')

    columns_raw = ['file',
                   'script_version_git',
                   'date',
                   'serum',
                   'particle',
                   'frame',
                   'y_micron',
                   'x_micron',
                   'diff_xy_micron',
                   'area_micron',
                   'perimeter_au_norm',
                   'min_dist_micron',
    ]

    canonical_fnames = {'diff_xy_micron': 'D',      # Locus displacement, microns
                        'area_micron': 'A',         # Nuclear area, square microns
                        'perimeter_au_norm': 'P',   # Nuclear perimeter, arbitrary units
                        'min_dist_micron': 'Dist',  # Minimal distance from locus to nuclear rim, microns
    }

    data_raw = pd.DataFrame(columns=columns_raw)
    
    for fpath in inp_path.rglob('*ome.tif'):
        filename = fpath.name

        (tiff, meta) = ome.read_ometiff(fpath)
        xy_pxsize = get_pxsize(meta)
        label_serum = int(filename.split('_')[1])

        chrom = rescale_intensity(tiff[:, 0, 0, :, :])
        lamin = rescale_intensity(tiff[:, 0, 1, :, :])

        chrom_r, masks, edges = regis(chrom, lamin, fast=False)
        data = find_dots(chrom_r)

        add_data = analyse_data(data, masks, edges, xy_pxsize, filename)
        add_data = add_xy_deltas(add_data)
        data_raw = data_raw.append(add_data, ignore_index=True, sort=False)

        data_raw[columns_raw] \
                .rename(columns=canonical_fnames) \
                .to_csv(out_path / 'tracking_raw.csv')


if __name__=="__main__":
    main()
