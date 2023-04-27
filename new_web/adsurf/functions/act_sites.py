from pymatgen.core.structure import IStructure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
import numpy as np

def get_act_sites(metal_poscar: str) -> dict:
    # TODO: Refine this function to find the active sites of different metal facets
    """Finds the active sites of a metal surface. These can be ontop, bridge or hollow sites.

    Parameters
    ----------
    metal_poscar : str
        Path to the POSCAR of the metal surface.

    Returns
    -------
    dict
        Dictionary with the atom indexes of each active site type.
    """
    surface = IStructure.from_file(metal_poscar)
    surf_sites = AdsorbateSiteFinder(surface, selective_dynamics=True)
    most_active_sites = surf_sites.find_adsorption_sites(put_inside=True)

    surface_sites = surf_sites.surface_sites

    act_sites_dict = {}
    for site in surface_sites:
        distances = np.linalg.norm(surface.cart_coords - site.coords, axis=1)
        index = np.argmin(distances)
        act_sites_dict[index] = surface[index].coords

    act_sites_arr = np.array([c for c in act_sites_dict.values()])
    
    active_site_dict = {'ontop':[], 'bridge':[], 'hollow':[]}
    for name_site, coords_site in most_active_sites.items():
        if name_site == 'all':
            continue

        for coord in coords_site:
            distances = np.linalg.norm(act_sites_arr - coord, axis=1)
            if name_site != 'hollow':
                index = np.argmin(distances)
                coords_in_surface = act_sites_arr[index]
                idx_surface = [key for key, coord in act_sites_dict.items() if np.all(coord == coords_in_surface)][0]

                active_site_dict[name_site].append(idx_surface)
            else:
                sorted_arr = np.sort(distances)[0:3]
                for i in sorted_arr:
                    coords_in_surface = act_sites_arr[np.where(distances == i)][0]
                    idx_surface = [key for key, coord in act_sites_dict.items() if np.all(coord == coords_in_surface)][0]
                    active_site_dict[name_site].append(idx_surface)

    return active_site_dict
