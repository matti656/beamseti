import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.ipac.ned import Ned
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from collections import defaultdict

def process_ned(
    dataframe,
    limit_distance_mpc=50,
    output_prefix=None,
    batch_size=10,
    max_workers=5,
    sleep_between_batches=5,
    H0=70.0,
    ri_method='median',
    split_by_band=False,
    n_shells=None,
    log_eirp_shells=None
):
    """
    Parallelized, batch-based NED extragalactic search and constraint pipeline for field-based SETI analysis.
    Designed to follow galaxy-targeted methods in Garrett & Siemion (2023) and Enriquez et al. (2018).
    
    Each field corresponds to a cone search around (RA, Dec) with given beam size.
    Detections from the NASA/IPAC Extragalactic Database (NED) are queried per beam.
    For each object, radiative constraints (EIRPmin), population estimates (N_stars), and Galaxy-type-dependent completeness corrections are computed.
    
    This function supports per-band and per-field batch analysis as in G&S 2023, with optional redshift-independent distance substitution (NED-D).
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain the following columns:
        - 'ra' [deg]: Right Ascension of beam center (ICRS, degrees), converted internally to float
        - 'dec' [deg]: Declination of beam center (ICRS, degrees), converted internally to float
        - 'fwhm_arcmin' [arcmin]: Full Width at Half Maximum of beam (Gaussians assumed), converted internally to float
        - 'fmin' [W * m^−2]: Band-dependent minimum detectable flux density, converted internally to float
        - 'nu_rel' [unitless]: total bandwidth of the receiver normalized by the central
            observing frequency (used in Transmitter Rate and CWTFM scaling), converted internally to float
        - 'field name': Unique field/mosaic name, converted to string internally
        - 'receiving_band': Observation band (e.g., 'L', 'S') used in multi-band analysis, converted to string internally
    
    limit_distance_mpc : float
        Maximum comoving distance (in Mpc) for catalog inclusion, e.g., 50 Mpc by default
    
    output_prefix : str or None, optional
        If provided, batch-level results and summary files are saved as CSV using this prefix
    
    batch_size : int, optional
        Number of fields grouped per processing batch (default: 10). Each batch is processed independently in parallel.
    
    max_workers : int, optional
        Number of parallel worker threads used for concurrent NED queries (default: 5).
    
    sleep_between_batches : float, optional
        Seconds to pause between launching new batches (avoids API throttling by NED)
    
    H0 : float, optional
        Hubble constant [km/s/Mpc] for converting redshift to distance if redshift-independent estimates unavailable (default: 70)
    
    ri_method : {'median', 'mean', 'all'}, optional
        Aggregation mode for NED-D redshift-independent distance estimates:
        - 'median': Use median of all NED-D distances per object
        - 'mean': Use arithmetic mean
        - 'all': Expand object into multiple entries, one per available distance
    
    split_by_band : bool, optional (default: False)
        If True, results are returned as per-receiving band dictionaries.
        If False, all fields are processed in one combined set.

    Add docstring for n_shells=None,
    log_eirp_shells=None, explaining that this affects the shell analysis (per object or per shell given)....


    Returns
    -------
    If split_by_band == False:
        ned_df : pd.DataFrame
            Concatenated table of all extragalactic objects queried across all fields, with beam scaling, frequency band, and per-object metadata.
        type_counts_df : pd.DataFrame
            A table of galaxy/object Type frequencies, useful for overview and diagnostics.
        summary_df : pd.DataFrame
            Table of unique extragalactic objects with:
            - EIRPmin limits [W]
            - Transmitter Rate log(TR) constraints
            - CWTFM values
        shell_results_df : pd.DataFrame
            Shell-based analysis for logEIRPmin sensitivity (see Gaia/SynthPop), with columns:
                - log_EIRPmin_shell
                - n_galaxies
                - N_stars
                - log_N_stars
                - log_N_stars_err
                - log_TR
                - log_TR_err
                - CWTFM
                - max_distance_mpc (of all objects in shell)
    
    If split_by_band == True:
        Returns dictionaries:
            ned_df_dict : {band_name: pd.DataFrame}
            type_counts_df_dict : {band_name: pd.DataFrame}
            summary_df_dict : {band_name: pd.DataFrame}
            shell_results_df_dict : {band_name: pd.DataFrame}
        Access each DataFrame by ned_df_dict['band 1']...,
        type_counts_df_dict['band 1']...
        summary_df_dict['band 1']...
        shell_results_df_dict['band 1']...
    Units
    -----
    - RA, Dec: degrees
    - fwhm_arcmin: arcminutes
    - fmin: Watts per meter squared [W * m^−2]
    - nu_rel(BW/f): unitless, where f represents the observed frequency, 
        and BW denotes the total bandwidth in GHz
    - distance_mpc: Mpc
    - scaling_factor: beam response [%] using Gaussian form: exp(−4ln(2) (θ / FWHM)^2)
    - EIRPmin: Watts [W]
    - log_EIRPmin: log10(W)
    - log_TR: log10(Transmitter Rate)
    - CWTFM: Continuous Waveform Transmitter Figure-of-Merit [unitless]
    
    Key Calculations (from G&S 2023) 
    -------------------------------
    - Beam response: Gaussian beam attenuation based on offset θ from center to object
    - EIRPmin:
        log₁₀(EIRPmin) = log₁₀(4π) + 2 log₁₀(d) + log₁₀(fmin) − log₁₀(response_frac)
        where d in meters, fmin in W * m^−2, response_frac is beam scaling factor (normalized based on beam center)
    - Transmitter Rate (TR):
        log_TR = − log₁₀(N_stars) − log₁₀(ν_rel)
    - N_stars:
        - Galaxy: 1e11 stars/galaxy assumed
        - Group/Cluster: Estimated from velocity dispersion, and if redshift NaN in NED-D and NED, then estimated from
            catalog designation, e.g., Clusters assigned 10 galaxies
        - Uncertainties propagated based on galaxy counts
    
    Notes
    -----
    - Results follow methodology introduced in Garrett & Siemion (2023)
    - Objects may appear multiple times across fields but are uniquely grouped for summary constraints.
    - Uses both redshift-independent distances (NED-D) and cosmological distances (cz / H0) where needed
    - Redshift limits applied: z ∈ [0.01, 0.1] for cz/H0 approximation to remain valid
    - Beam falloff is computed using the angular offset and input FWHM in arcminutes
    - Each receiving band is expected to have uniform ν_rel and fmin values across fields (used for CWTFM)
    - Summary table removes duplicates and produces one constraint per extragalactic object
    
    Citations
    ---------
    - M A Garrett, A P V Siemion, Constraints on extragalactic transmitters via Breakthrough Listen observations 
        of background sources, Monthly Notices of the Royal Astronomical Society, Volume 519, Issue 3, March 2023, 
        Pages 4581–4588, https://doi.org/10.1093/mnras/stac2607
    - B S Wlodarczyk-Sroka, M A Garrett, A P V Siemion, Extending the Breakthrough Listen nearby star survey to 
        other stellar objects in the field, Monthly Notices of the Royal Astronomical Society, Volume 498, Issue 4, 
        November 2020, Pages 5720–5729, https://doi.org/10.1093/mnras/staa2672
    """
    # ... [all previous code up to summary_df creation, i.e. do not modify ned_batch_query or ned_catalog_summary] ...
    dataframe = dataframe.copy()
    dataframe.loc[:, 'field name'] = dataframe.loc[:, 'field name'].astype(str)
    dataframe.loc[:, 'receiving_band'] = dataframe.loc[:, 'receiving_band'].astype(str)
    dataframe.loc[:, 'ra'] = dataframe.loc[:, 'ra'].astype(float)
    dataframe.loc[:, 'dec'] = dataframe.loc[:, 'dec'].astype(float)
    dataframe.loc[:, 'fwhm_arcmin'] = dataframe.loc[:, 'fwhm_arcmin'].astype(float)
    dataframe.loc[:, 'fmin'] = dataframe.loc[:, 'fmin'].astype(float)
    dataframe.loc[:, 'nu_rel'] = dataframe.loc[:, 'nu_rel'].astype(float)

    def cluster_nstar_error(n_gal):
            if n_gal < 10:
                return 0.3
            return 0.5 + 0.2 * max(0, np.floor(np.log10(n_gal)) - 1)
        
    # Batch-cone-search and combine results for a given fields subset
    def ned_batch_query(fields_df, band_name, output_prefix=None):
        n_total = len(fields_df)
        batch_files = []

        def angular_separation_arcmin(beam_ra, beam_dec, star_ra, star_dec):
            c1 = SkyCoord(beam_ra, beam_dec, unit='deg')
            c2 = SkyCoord(star_ra, star_dec, unit='deg')
            return c1.separation(c2).arcminute

        def beam_scaling_percent(beam_ra, beam_dec, star_ra, star_dec, fwhm_arcmin):
            theta_arcmin = angular_separation_arcmin(beam_ra, beam_dec, star_ra, star_dec)
            exponent = -4 * np.log(2) * (theta_arcmin / fwhm_arcmin)**2
            scaling_factor = np.exp(exponent) * 100  # percent
            return theta_arcmin, scaling_factor

        def process_beam_vectorized(beam_ra, beam_dec, fwhm_arcmin, catalog_df):
            star_ras = catalog_df['ra'].to_numpy()
            star_decs = catalog_df['dec'].to_numpy()
            theta, scaling = beam_scaling_percent(beam_ra, beam_dec, star_ras, star_decs, fwhm_arcmin)
            catalog_df = catalog_df.copy()
            catalog_df['theta_arcmin'] = theta
            catalog_df['scaling_factor'] = scaling
            return catalog_df

        def fetch_ned_batch(batch_indices, batch_idx):
            batch_results = []
            for i in batch_indices:
                row = fields_df.iloc[i]
                ra = float(row['ra'])
                dec = float(row['dec'])
                fwhm = float(row['fwhm_arcmin'])
                field_name = str(row['field name'])
                fmin = float(row['fmin'])
                nu_rel = float(row['nu_rel'])
                try:
                    result_table = Ned.query_region(
                        SkyCoord(ra=ra * u.deg, dec=dec * u.deg),
                        radius=(fwhm / 2.0) * u.arcmin)
                    df = result_table.to_pandas()
                    df = df.rename(columns={"RA": "ra", "DEC": "dec"})
                    df['field_name'] = field_name
                    df['fmin'] = fmin
                    df['nu_rel'] = nu_rel
                    df['fwhm_arcmin'] = fwhm
                    df = process_beam_vectorized(ra, dec, fwhm, df)
                    batch_results.append(df)
                except Exception as e:
                    print(f"Error querying NED for RA={ra}, Dec={dec}, FWHM={fwhm}: {e}")
            if batch_results:
                batch_df = pd.concat(batch_results, ignore_index=True)
                # if output_prefix:
                #     filename = f"{output_prefix}_NED_{band_name}_batch{batch_idx}.csv"
                #     batch_df.to_csv(filename, index=False)
                #     return filename
                # else:
                return batch_df
            return None

        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                batch_indices = list(range(batch_start, batch_end))
                futures.append(executor.submit(fetch_ned_batch, batch_indices, batch_start // batch_size))
                time.sleep(sleep_between_batches)
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    batch_results.append(res)

        if not batch_results:
            print("No NED batch results fetched.")
            return pd.DataFrame()

        # if output_prefix:
        #     dfs = [pd.read_csv(f) for f in batch_results]
        # else:
        dfs = batch_results
        ned_df = pd.concat(dfs, ignore_index=True)
        # Clean up columns
        if 'eirp_boost' in ned_df.columns:
            ned_df = ned_df.drop(columns=['eirp_boost'])
        for c in ['input_ra', 'input_dec', 'input_fwhm_arcmin', 'd_max_mpc']:
            if c in ned_df.columns:
                ned_df = ned_df.drop(columns=c)
        ned_df.drop_duplicates(subset=['Object Name'], inplace=True)
        col_to_move = ned_df.pop('field_name')
        ned_df.insert(0, 'field_name', col_to_move)
        if output_prefix:
            ned_df.to_csv(f"{output_prefix}_ned_df.csv", index=False)
        type_col = 'Type' if 'Type' in ned_df.columns else 'type'
        unique_types, type_counts = np.unique(ned_df[type_col].fillna('Unknown'), return_counts=True)
        type_counts_df = pd.DataFrame({'Type': unique_types, 'Count': type_counts})
        if output_prefix:
            type_counts_df.to_csv(f"{output_prefix}_type_counts_df.csv", index=False)
        return ned_df, type_counts_df

    # Catalog summary (deduplication, grouping, EIRPmin/constraints)
    def ned_catalog_summary(
        ned_df,
        limit_distance_mpc,
        output_prefix=None,
        H0=70.0,
        ri_method='median',
    ):
        def get_ned_redshift_independent_distances():
            url = "https://ned.ipac.caltech.edu/Archive/Distances/NED30.5.1-D-17.1.2-20200415.csv"
            column_names = [
                "Exclusion Code", "Record index", "Object index", "Galaxy ID",
                "m-M", "err", "D (Mpc)", "Method", "REFCODE", "SN ID",
                "redshift (z)", "Hubble const.", "Adopted LMC modulus",
                "Date (Yr. - 1980)", "Notes"
            ]
            ned_data = pd.read_csv(url, names=column_names, header=None, low_memory=False)
            return ned_data

        ned_df = ned_df.copy()
        ned_df['field name'] = ned_df['field_name'].astype(str)
        ned_data = get_ned_redshift_independent_distances()

        ri_dist_dict = defaultdict(list)
        ri_z_dict = defaultdict(list)
        for _, row in ned_data.iterrows():
            galid = str(row['Galaxy ID']).strip()
            dmpc = pd.to_numeric(row['D (Mpc)'], errors='coerce')
            zri = pd.to_numeric(row['redshift (z)'], errors='coerce')
            if not pd.isnull(dmpc):
                ri_dist_dict[galid].append(dmpc)
            if not pd.isnull(zri):
                ri_z_dict[galid].append(zri)
        # Add/combine NED-D for all objects
        expanded_rows = []
        for _, row in ned_df.iterrows():
            obj = row['Object Name']
            has_ri_dist = obj in ri_dist_dict and len(ri_dist_dict[obj]) > 0
            has_ri_z = obj in ri_z_dict and len(ri_z_dict[obj]) > 0
            if has_ri_dist or has_ri_z:
                dvals = ri_dist_dict[obj] if has_ri_dist else []
                zvals = ri_z_dict[obj] if has_ri_z else []
                if ri_method == "median":
                    dval = np.median(dvals) if dvals else np.nan
                    derr = 0.741 * (np.percentile(dvals, 75)-np.percentile(dvals, 25)) if len(dvals)>1 else np.nan
                    zval = np.median(zvals) if zvals else np.nan
                    new_row = row.copy()
                    new_row['distance_mpc'], new_row['distance_mpc_err'] = dval, derr
                    new_row['Redshift'] = zval if not pd.isnull(zval) else row['Redshift']
                    new_row['distance_method'] = 'NED-RI Median'
                    expanded_rows.append(new_row)
                elif ri_method == "mean":
                    dval = np.mean(dvals) if dvals else np.nan
                    derr = np.std(dvals) if len(dvals)>1 else np.nan
                    zval = np.mean(zvals) if zvals else np.nan
                    new_row = row.copy()
                    new_row['distance_mpc'], new_row['distance_mpc_err'] = dval, derr
                    new_row['Redshift'] = zval if not pd.isnull(zval) else row['Redshift']
                    new_row['distance_method'] = 'NED-RI Mean'
                    expanded_rows.append(new_row)
                elif ri_method == "all":
                    L = max(len(dvals), len(zvals), 1)
                    for k in range(L):
                        new_row = row.copy()
                        new_row['distance_mpc'] = dvals[k] if k < len(dvals) else np.nan
                        new_row['distance_mpc_err'] = np.nan
                        new_row['Redshift'] = zvals[k] if k < len(zvals) and not pd.isnull(zvals[k]) else row['Redshift']
                        new_row['distance_method'] = 'NED-RI Single'
                        expanded_rows.append(new_row)
                else:
                    expanded_rows.append(row)
            else:
                expanded_rows.append(row)
        ned_df = pd.DataFrame(expanded_rows)
        if 'distance_mpc' in ned_df:
            ned_d_df = ned_df[ned_df['distance_mpc'].notnull()]
            print(len(ned_d_df))
        # Assign cz/H0 distance, if NED-D absent
        c_kms = 299792.458
        def assign_distance(row):
            if 'distance_mpc' in row and not pd.isnull(row['distance_mpc']):
                return row
            z = row.get('Redshift', np.nan)
            if pd.isnull(z) or z < 0 or z < 0.01 or z >= 0.1:
                row['distance_mpc'] = np.nan
                return row
            row['distance_mpc'] = (z * c_kms) / H0
            return row
        ned_df = ned_df.apply(assign_distance, axis=1)
        # Membership, deduplication, grouping
        redshift_window_params = dict(pair=500, triple=600, group=1000, cluster=1500)
        group_names = {'pair': 'GPair', 'triple': 'GTrpl', 'group': 'GGroup', 'cluster': 'GClstr'}
        group_membership = defaultdict(set)
        for kind, tag in group_names.items():
            disp = redshift_window_params[kind]
            groups = ned_df[(ned_df['Type'] == tag) & (ned_df['Redshift'].notnull())]
            for _, group in groups.iterrows():
                z0, ra0, dec0 = group['Redshift'], group['ra'], group['dec']
                dz = disp / c_kms
                zmin, zmax = z0 - dz, z0 + dz
                radius_deg = group.get('fwhm_arcmin', 3.0) / 2.0 / 60.0
                gcenter = SkyCoord(ra0 * u.deg, dec0 * u.deg)
                valid = (
                    ned_df['ra'].notnull() & ned_df['dec'].notnull() &
                    np.isfinite(ned_df['ra']) & np.isfinite(ned_df['dec'])
                )
                if not valid.all():
                    n_bad = (~valid).sum()
                    print(f"Warning: Removing {n_bad} objects with invalid RA/DEC for SkyCoord.")
                # If you only want to filter for the SkyCoord step:
                ra_arr = ned_df.loc[valid, 'ra'].astype(float).values
                dec_arr = ned_df.loc[valid, 'dec'].astype(float).values
                all_coords = SkyCoord(ra_arr * u.deg, dec_arr * u.deg)
                separation = gcenter.separation(all_coords).deg
                members = ned_df[
                    (separation <= radius_deg) &
                    (ned_df['Redshift'] >= zmin) & (ned_df['Redshift'] <= zmax) &
                    (ned_df['Type'] == 'G')
                ]
                member_names = set(members['Object Name'])
                group_membership[group['Object Name']] |= member_names

        # def cluster_nstar_error(n_gal):
        #     if n_gal < 10:
        #         return 0.3
        #     return 0.5 + 0.2 * max(0, np.floor(np.log10(n_gal)) - 1)

        def get_nstars(row):
            obj = row.get('Object Name')
            t = row.get('Type', '')
            if obj in group_membership and len(group_membership[obj]) > 0:
                n_gal = len(group_membership[obj])
                err = cluster_nstar_error(n_gal)
                return pd.Series({'N_stars': n_gal * 1e11,
                                  'log_N_stars': np.log10(n_gal * 1e11),
                                  'log_N_stars_err': err})
            elif t in ('G_Lens', 'GPair', 'GTrpl', 'GGroup', 'GClstr'):
                n_gal = {'G_Lens': 1, 'GPair': 2, 'GTrpl': 3, 'GGroup': 4, 'GClstr': 10}.get(t, 1)
                err = cluster_nstar_error(n_gal)
                return pd.Series({'N_stars': n_gal * 1e11, 'log_N_stars': np.log10(n_gal * 1e11), 'log_N_stars_err': err})
            else:
                return pd.Series({'N_stars': 1e11, 'log_N_stars': 11., 'log_N_stars_err': 0.3})
        nstars_cols = ned_df.apply(get_nstars, axis=1)
        for c in nstars_cols.columns: ned_df[c] = nstars_cols[c]

        d_m = ned_df['distance_mpc'].astype(float) * 3.086e22
        log4pi = np.log10(4 * np.pi)
        scaling_fraction = ned_df['scaling_factor'] / 100.0
        with np.errstate(invalid='ignore', divide='ignore'):
            ned_df['logEIRPmin'] = log4pi + 2 * np.log10(d_m) + np.log10(ned_df['fmin']) - np.log10(scaling_fraction)
            ned_df['EIRPmin'] = 10 ** ned_df['logEIRPmin']
            ned_df['log_Transmitter_Rate'] = - ned_df['log_N_stars'] - np.log10(ned_df['nu_rel'])
            ned_df['CWTFM'] = (500 / 1e13) * ned_df['EIRPmin'] / (ned_df['N_stars'] * ned_df['nu_rel'])
        # Unique summary (all types)
        unique_objects = set(ned_df.loc[ned_df['distance_mpc'].notnull(), 'Object Name'])
        for members in group_membership.values(): unique_objects.update(members)
        summary_rows = []
        for obj in unique_objects:
            obj_rows = ned_df[(ned_df['Object Name'] == obj) & (ned_df['distance_mpc'].notnull())]
            if obj_rows.empty: continue
            row = obj_rows.iloc[0]
            N_stars, log_N_stars, log_N_stars_err = row['N_stars'], row['log_N_stars'], row['log_N_stars_err']
            median_dist = obj_rows['distance_mpc'].median()
            avg_beam_response = obj_rows['scaling_factor'].mean()
            median_dist_m = median_dist * 3.086e22 if not np.isnan(median_dist) else np.nan
            if np.isnan(median_dist) or np.isnan(avg_beam_response) or avg_beam_response <= 0:
                log_EIRPmin = np.nan; CWTFM = np.nan
            else:
                log_EIRPmin = log4pi + 2 * np.log10(median_dist_m) + np.log10(row['fmin']) - np.log10(avg_beam_response / 100)
                EIRPmin = 10 ** log_EIRPmin
                CWTFM = (500 / 1e13) * EIRPmin / (N_stars * row['nu_rel'])
            log_TR = -log_N_stars - np.log10(row['nu_rel']) if N_stars > 0 else np.nan
            log_TR_err = log_N_stars_err
            summary_rows.append({
                'field_name': row['field name'],
                'object_name': obj,
                'type': row['Type'],
                'distance_mpc': median_dist,
                'offset_arcmin': obj_rows['theta_arcmin'].mean(),
                'beam_response_percent': avg_beam_response,
                'nu_rel': row['nu_rel'],
                'N_stars': N_stars,
                'log_Nstars': log_N_stars,
                'log_Nstars_err': log_N_stars_err,
                'log_EIRPmin': log_EIRPmin,
                'log_TR': log_TR,
                'log_TR_err': log_TR_err,
                'CWTFM': CWTFM
            })
        summary_df = pd.DataFrame(summary_rows)
        if summary_df.empty or 'distance_mpc' not in summary_df.columns:
            if output_prefix:
                summary_df.to_csv(f"{output_prefix}_NED_catalog_summary.csv", index=False)
            return summary_df
        summary_df = summary_df[(summary_df['distance_mpc'].notnull()) & (summary_df['distance_mpc'] <= limit_distance_mpc)]
        if output_prefix:
            summary_df.to_csv(f"{output_prefix}_NED_catalog_summary.csv", index=False)
        return summary_df
    
    def ned_shell_summary(
        summary_df,
        limit_distance_mpc,
        n_shells=None,
        log_eirp_shells=None,
        output_prefix=None
    ):
        # Defensive programming to avoid KeyError
        if summary_df.empty or 'distance_mpc' not in summary_df.columns or 'log_EIRPmin' not in summary_df.columns:
            print('Warning: summary_df is empty or missing key columns.')
            if output_prefix:
                shell_results_df = pd.DataFrame()
                shell_results_df.to_csv(f"{output_prefix}_NED_shell_results.csv", index=False)
            return pd.DataFrame()
        # For each unique object, we already have logEIRPmin and N_stars, log_Nstars, log_Nstars_err
        valid_df = summary_df[summary_df['distance_mpc'].notnull() & (summary_df['distance_mpc'] <= limit_distance_mpc) & summary_df['log_EIRPmin'].notnull()]
        # Choose shell grid
        if log_eirp_shells is not None:
            log_eirp_shells_to_use = np.array(log_eirp_shells)
        elif n_shells is not None and n_shells > 1:
            log_min = valid_df['log_EIRPmin'].min()
            log_max = valid_df['log_EIRPmin'].max()
            log_eirp_shells_to_use = np.linspace(np.floor(log_min), np.ceil(log_max), n_shells)
        else:  # Default: use every unique value as a shell edge
            log_eirp_shells_to_use = np.sort(valid_df['log_EIRPmin'].dropna().unique())
        results = []
        for log_shell_val in log_eirp_shells_to_use:
            # Mask for objects within shell
            mask = valid_df['log_EIRPmin'] <= log_shell_val
            shell_objects = valid_df[mask]
            # n_obj = len(shell_objects)
            # Total stars is sum of stars from each object within shell
            tot_nstars = shell_objects['N_stars'].sum()
            log_nstars = np.log10(tot_nstars) if tot_nstars > 0 else np.nan
            n_gal = tot_nstars / (1e11)
            err = cluster_nstar_error(n_gal)
            # log_N_stars_err
            log_nstars_err = err
            # log_TR uses tot_nstars and nu_rel (mean)
            nu_rel_shell = shell_objects['nu_rel'].mean() if n_gal > 0 else np.nan
            logTR = - log_nstars - np.log10(nu_rel_shell) if (tot_nstars > 0 and nu_rel_shell > 0) else np.nan
            logTR_err = log_nstars_err  # By design for NED
            # CWTFM (like elsewhere)
            CWTFM = (log_shell_val / 1e13) * (0.5 / nu_rel_shell) * (1000 / tot_nstars) if (tot_nstars > 0 and nu_rel_shell > 0) else np.nan
            # Max distance among objects in shell
            max_dist_mpc = shell_objects['distance_mpc'].max() if n_gal > 0 else np.nan
            results.append({
                'log_EIRPmin_shell': log_shell_val,
                'nu_rel': nu_rel_shell,
                'n_galaxies': n_gal,
                'N_stars': tot_nstars,
                'log_N_stars': log_nstars,
                'log_N_stars_err': log_nstars_err,
                'log_TR': logTR,
                'log_TR_err': logTR_err,
                'CWTFM': CWTFM,
                'max_distance_mpc': max_dist_mpc
            })
        shell_results_df = pd.DataFrame(results)
        if output_prefix:
            shell_results_df.to_csv(f"{output_prefix}_NED_shell_results.csv", index=False)
        return shell_results_df
        
    if split_by_band:
        ned_df_dict = {}
        type_counts_df_dict = {}
        summary_df_dict = {}
        shell_results_dict = {}
        for band in dataframe['receiving_band'].unique():
            band_df = dataframe[dataframe['receiving_band'] == band].reset_index(drop=True)
            print(f"Processing receiving_band={band} with {len(band_df)} fields.")
            out_prefix = f"{output_prefix}_{band}" if output_prefix else None
            ned_df, type_counts_df = ned_batch_query(band_df, band, out_prefix)
            summary_df = ned_catalog_summary(
                ned_df,
                limit_distance_mpc=limit_distance_mpc,
                output_prefix=out_prefix,
                H0=H0,
                ri_method=ri_method,
            )
            # Produce shell summary for this band
            shell_results_df = ned_shell_summary(
                summary_df, limit_distance_mpc,
                n_shells=n_shells, log_eirp_shells=log_eirp_shells,
                output_prefix=out_prefix
            )
            ned_df_dict[band] = ned_df
            type_counts_df_dict[band] = type_counts_df
            summary_df_dict[band] = summary_df
            shell_results_dict[band] = shell_results_df
        return ned_df_dict, type_counts_df_dict, summary_df_dict, shell_results_dict
    else:
        ned_df, type_counts_df = ned_batch_query(dataframe, "all", output_prefix)
        summary_df = ned_catalog_summary(
            ned_df,
            limit_distance_mpc=limit_distance_mpc,
            output_prefix=output_prefix,
            H0=H0,
            ri_method=ri_method,
        )
        shell_results_df = ned_shell_summary(
            summary_df, limit_distance_mpc,
            n_shells=n_shells, log_eirp_shells=log_eirp_shells,
            output_prefix=output_prefix
        )
        return ned_df, type_counts_df, summary_df, shell_results_df
