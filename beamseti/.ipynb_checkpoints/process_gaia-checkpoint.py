import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_gaia(
    dataframe,
    limit_distance_pc=10000,
    output_prefix=None,
    batch_size=10,
    max_workers=5,
    sleep_between_batches=5,
    n_shells=8,
    log_eirp_shells=None,
    split_by_band=False
):
    """
    Remember to login to the Gaia archive before using this function!
    Gaia.login(user='userName', password='userPassword')
        
    Parallelized, batch-based Gaia cone search and post-processing pipeline for field-based SETI analysis,
    following the methodology of Wlodarczyk-Sroka et al. (2020, MNRAS, 498, 5720) and incorporating
    distance estimates from Bailer-Jones et al. (2021, AJ, 161, 147, Gaia EDR3).
    
    This function enables automated, scalable querying of the Gaia EDR3 catalog with rigorous field geometry,
    beam attenuation, and EIRPmin constraint analysis per observing field and receiving band, allowing both
    combined (all bands) and per-band results as in recent large-scale SETI surveys.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Required columns:
        - 'ra'           [degrees]: ICRS Right Ascension (field/beam center, converted internally to float)
        - 'dec'          [degrees]: ICRS Declination (field/beam center, converted internally to float)
        - 'fwhm_arcmin'  [arcminutes]: Beamwidth (FWHM) of the field, converted internally to float
        - 'fmin'         [W * m^−2]: Minimum detectable flux density at the field (per band), converted internally to float
        - 'nu_rel'       [unitless]: total bandwidth of the receiver normalized by the central, converted internally to float
            observing frequency (used in Transmitter Rate and CWTFM scaling)
        - 'field name': Unique field identifier, converted internally to str
        - 'receiving_band': Name of the observing band (e.g., 'L', 'S'); allows per-band separation, converted internally to str
    
        Acceptable types: all columns must be convertible to their indicated types.
    
    limit_distance_pc : float, optional
        Maximum stellar distance (parsecs) to include from Gaia (default: 10 kpc, as in WG&S 2020)
    output_prefix : str, optional
        If provided, concatenated results are saved in CSV files using this prefix (per band, or combined)
        all_df is saved according to output_prefix_band name_gaia_df.csv
        shell_results is saved according to output_prefix_band name_gaia_shell_results.csv
    batch_size : int, optional
        Number of fields to process in each Gaia ADQL/cone search batch (default: 10, scale up as needed)
    max_workers : int, optional
        Maximum number of concurrent batch query threads (default: 5, scale up as needed)
    sleep_between_batches : float, optional
        Seconds to pause between launching concurrent batches (default: 5, scale down as needed)
    n_shells : int, optional
        Number of log10(EIRPmin) shells, if not providing log_eirp_shells explicitly (default: 8 as per WG&S 2020)
    log_eirp_shells : array-like, optional
        If specified, manual edges (log₁₀ EIRPmin [W]) for sensitivity shell analysis. Per WG&S 2020, for max distances less 
        than 10 kpc the standard range of EIRPmin shells is np.linspace(11, 18, 8)
    split_by_band : bool, optional
        If True, runs the complete analysis per unique value of 'receiving_band' (returns dict per band).
        If False, combines all bands in the input for a single global analysis.
    
    Returns
    -------
    If split_by_band == False:
        all_df : pandas.DataFrame
            Detected Gaia sources with beam geometry/attenuation factors, joined with input field metadata.
            Columns include:
                - source_id            [int]
                - ra, dec              [deg]
                - r_med_geo, r_lo_geo, r_hi_geo    [pc; median/lo/hi geometric distance]
                - phot_g_mean_mag, bp_rp, abs_g_photogeo, abs_g_geo [mag]
                - field_name           [str]
                - fmin, nu_rel, fwhm_arcmin
                - theta_arcmin         [arcmin]: offset from field center (beam attenuation)
                - scaling_factor       [unitless]: Gaussian beam response normalized based on beam center
        shell_results : pandas.DataFrame
            Table of EIRPmin shells:
                - log_EIRPmin_shell    [log10(W)] (shell edges)
                - n_stars              (stars at EIRPmin ≤ shell edge)
                - n_stars_pos_err, n_stars_neg_err (uncertainty based on distance bounds)
                - log_TR, log_TR_pos_err, log_TR_neg_err
                - CWTFM, CWTFM_pos_err, CWTFM_neg_err
                - max_distance_pc       [pc]: maximal distance for each shell
    
    If split_by_band == True:
        all_df_dict : dict
            Per-band DataFrame for each unique 'receiving_band'
        shell_results_dict : dict
            Per-band shell analysis DataFrame
        Access each DataFrame by all_df_dict['band 1']...,
            shell_results_dict['band 1']...
    
    Methods & Variables (per WG&S 2020; Bailer-Jones et al. 2021)
    -------------------------------------------------------------
    - Stellar selection/search: Each field is queried via Gaia EDR3, retrieving all sources within
      the primary beam FWHM and satisfying ruwe < 1.4 (astrometric quality flag). Fields are processed
      in batches for efficiency using multi-threading (see §2.3 of WG&S 2020).
    - Distance estimation: Stellar distances taken from Bailer-Jones et al. 2021 (geometric and photogeometric), columns r_med_geo, r_lo_geo, r_hi_geo [pc].
    - Beam response calculation: For each source, the beam attenuation is
        scaling_factor = exp(−4 ln(2) * (θ/FWHM)^2)
        where θ is the angular separation from beam center (arcmin), and FWHM is the standard field width in arcmin where beam response drops to 50% 
        (WG&S Eq. 1 and §2.2).
    - Minimum detectable EIRP: For each star and each shell:
          log₁₀(EIRPmin) = log₁₀(4π) + 2 log₁₀(distance_m) + log₁₀(fmin) – log₁₀(beam_response)
      where distance_m is in meters (1 pc = 3.086e16 m), fmin in W * m^-2.
    - Sensitivity shells: log10(EIRPmin) shells are constructed either manually, or using
        np.linspace(11, 18, 8) for distances < 10kpc (WG&S Table 2).
    - Error propagation: For each shell, uncertainties in distance (lo/hi) propagate into star counts and log_TR, CWTFM, 
        and their errors. Shell statistics (n_stars, log_TR, etc.) include star-count errors where postive error in n_stars (n_stars_pos_error) comes
        from r_med_geo > shell value, and r_lo_geo <= shell value. Negative error in star-count (n_stars_neg_error) comes from r_med_geo <= shell value, and 
        r_hi_geo > shell value. Log_TR_pos_err, log_TR_neg_err, CWTFM_pos_err, CWTFM_neg_err all come from the difference between the nominal values
        evaluated at r_med_geo and the upper and lower values of n_stars(n_stars_upper and n_stars_lower), 
        where n_stars_upper = n_stars + n_stars_pos_error and 
        n_stars_lower = n_stars - n_stars_neg_error.
    - Population constraint metrics:
        - log_TR: log₁₀(1 / (N_stars * nu_rel)), where N_stars is the number of stars below the shell EIRPmin value (WG&S Eq. 3).
        - CWTFM (Continuous Waveform Transmitter Figure of Merit):
          (shell EIRPmin / 1e13) * (0.5 / nu_rel) * (1000 / N_stars)
    
    Units
    -----
    - RA, Dec: degrees [deg]
    - fwhm_arcmin: arcminutes [']
    - distance: parsecs [pc] (converted to meters [m] for EIRPmin calculation)
    - fmin: [W * m^-2]
    - nu_rel: nu_rel(BW/f): unitless, where f represents the observed frequency, 
        and BW denotes the total bandwidth in GHz
    - scaling_factor: unitless (0.5 ≤ scaling factor ≤ 1)
    - theta_arcmin: arcminutes [']
    - log_EIRPmin: log₁₀(W)
    - log_TR: log₁₀(1/(N_stars * nu_rel)) [dimensionless]
    - CWTFM: unitless
    
    Notes
    -----
    - The 'field name' and 'receiving_band' columns are always mapped to string dtype internally. All other columns are converted to floats.
    - All results, per-band or combined, ensure unique source_id entries (no duplicate Gaia sources per band).
    - Any column with missing/failed data (e.g., stars without BJ21 distance) is excluded from shell analysis.
    - If split_by_band is True, output dicts can be accessed by their receiving_band name key (e.g., all_df_dict['L']).
    
    References
    ----------
    - Wlodarczyk-Sroka, B. S., Garrett, M. A., Siemion, A. P. V. (2020), "Extending the Breakthrough Listen nearby star survey 
        to other stellar objects in the field," MNRAS, 498, 5720. https://doi.org/10.1093/mnras/staa2672
    - Bailer-Jones, C.A.L., et al. (2021), "Estimating Distances from Parallaxes. V. Geometric and Photogeometric Distances to 
        1.47 Billion Stars in Gaia Early Data Release 3," AJ, 161, 147. https://iopscience.iop.org/article/10.3847/1538-3881/abd806
    - See also SETI EIRPmin and shell statistics methodology as adopted in Gajjar & Siemion (2023), Enriquez et al. (2018), and relatedSETI literature.
    """
    dataframe = dataframe.copy()
    dataframe.loc[:, 'field name'] = dataframe.loc[:, 'field name'].astype(str)
    dataframe.loc[:, 'receiving_band'] = dataframe.loc[:, 'receiving_band'].astype(str)
    dataframe.loc[:, 'ra'] = dataframe.loc[:, 'ra'].astype(float)
    dataframe.loc[:, 'dec'] = dataframe.loc[:, 'dec'].astype(float)
    dataframe.loc[:, 'fwhm_arcmin'] = dataframe.loc[:, 'fwhm_arcmin'].astype(float)
    dataframe.loc[:, 'fmin'] = dataframe.loc[:, 'fmin'].astype(float)
    dataframe.loc[:, 'nu_rel'] = dataframe.loc[:, 'nu_rel'].astype(float)

    def run_pipeline(df_band, band_name, output_prefix=None, 
    log_eirp_shells_outer=None, n_shells_outer=None):

        log_eirp_shells_local = log_eirp_shells_outer
        n_shells_local = n_shells_outer
        n_total = len(df_band)
        batch_files = []

        def angular_separation_arcmin(ra1, dec1, ra2, dec2):
            c1 = SkyCoord(ra1, dec1, unit='deg')
            c2 = SkyCoord(ra2, dec2, unit='deg')
            return c1.separation(c2).arcminute

        def beam_scaling(beam_ra, beam_dec, star_ra, star_dec, fwhm_arcmin):
            theta_arcmin = angular_separation_arcmin(beam_ra, beam_dec, star_ra, star_dec)
            exponent = -4 * np.log(2) * (theta_arcmin / fwhm_arcmin)**2
            scaling_factor = np.exp(exponent)
            return theta_arcmin, scaling_factor

        def process_beam_astropy_vectorized(beam_ra, beam_dec, fwhm_arcmin, catalog_df):
            star_ras = catalog_df['ra'].to_numpy()
            star_decs = catalog_df['dec'].to_numpy()
            theta, scaling = beam_scaling(beam_ra, beam_dec, star_ras, star_decs, fwhm_arcmin)
            catalog_df = catalog_df.copy()
            catalog_df.loc[:, 'theta_arcmin'] = theta
            catalog_df.loc[:, 'scaling_factor'] = scaling
            return catalog_df
        
        def build_multi_cone_adql(ra_list, dec_list, fwhm_list, field_name_list):
            queries = []
            for idx, (ra, dec, fwhm, fname) in enumerate(zip(ra_list, dec_list, fwhm_list, field_name_list)):
                radius_deg = (fwhm / 2.0) / 60.0
                queries.append(f"""
                SELECT
                  source_id, ra, dec,
                  r_med_geo, r_lo_geo, r_hi_geo,
                  r_med_photogeo, r_lo_photogeo, r_hi_photogeo,
                  phot_bp_mean_mag - phot_rp_mean_mag AS bp_rp,
                  phot_g_mean_mag,
                  phot_g_mean_mag - 5 * LOG10(r_med_geo) + 5 AS abs_g_geo,
                  phot_g_mean_mag - 5 * LOG10(r_med_photogeo) + 5 AS abs_g_photogeo,
                  '{fname}' AS field_name
                FROM gaiaedr3.gaia_source
                JOIN external.gaiaedr3_distance USING (source_id)
                WHERE
                  ruwe < 1.4 AND
                  1 = CONTAINS(
                    POINT('ICRS', {ra}, {dec}),
                    CIRCLE('ICRS', ra, dec, {radius_deg})
                  )
                """)
            return " UNION ALL ".join(queries)

        def fetch_batch(ra_batch, dec_batch, fwhm_batch, field_name_batch, batch_idx):
            adql = build_multi_cone_adql(ra_batch, dec_batch, fwhm_batch, field_name_batch)
            job = Gaia.launch_job_async(adql)
            result = job.get_results().to_pandas()
            result = result.drop_duplicates(subset=['source_id'])
            for ra, dec, fwhm, fname in zip(ra_batch, dec_batch, fwhm_batch, field_name_batch):
                mask = result['field_name'] == fname
                if mask.any():
                    subset = result.loc[mask]
                    processed = process_beam_astropy_vectorized(ra, dec, fwhm, subset)
                    processed = processed.loc[subset.index]
                    result.loc[mask, ['theta_arcmin', 'scaling_factor']] = processed[['theta_arcmin', 'scaling_factor']].to_numpy()
            filtered_df = result[(result['r_med_geo'] <= limit_distance_pc) & (result['r_med_geo'] >= 0)].copy()
            # if output_prefix is not None:
                # filename = f"{output_prefix}_{band_name}_batch{batch_idx}.csv"
                # filtered_df.to_csv(filename, index=False)
            return filtered_df

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                ra_batch = df_band['ra'].iloc[batch_start:batch_end].to_numpy()
                dec_batch = df_band['dec'].iloc[batch_start:batch_end].to_numpy()
                fwhm_batch = df_band['fwhm_arcmin'].iloc[batch_start:batch_end].to_numpy()
                field_name_batch = df_band['field name'].iloc[batch_start:batch_end].to_numpy()
                futures.append(executor.submit(
                    fetch_batch, ra_batch, dec_batch, fwhm_batch, field_name_batch, batch_start // batch_size
                ))
                time.sleep(sleep_between_batches)
            dfs = []
            for future in as_completed(futures):
                dfb = future.result()
                dfs.append(dfb)
            if not dfs:
                return pd.DataFrame(), pd.DataFrame()
            all_df = pd.concat(dfs, ignore_index=True)
            all_df['field_name'] = all_df['field_name'].astype(str)
            for col in ['theta_arcmin', 'scaling_factor']:
                if col not in all_df.columns:
                    all_df[col] = np.nan
            input_field_map = df_band.set_index('field name')[['fmin','nu_rel','fwhm_arcmin']].copy()
            all_df = all_df.merge(input_field_map, left_on='field_name', right_index=True, how='left')
            # Move field_name to first column
            col_to_move = all_df.pop('field_name')
            all_df.insert(0, 'field_name', col_to_move)
            all_df.drop_duplicates(subset=['source_id'], inplace=True)
        
        # EIRPmin + Shell Analysis
        def calculate_eirpmin_log(df):
            df = df.copy()
            log_4pi = np.log10(4 * np.pi)
            for key, col in zip(['med', 'lo', 'hi'], ['r_med_geo', 'r_lo_geo', 'r_hi_geo']):
                d_m = df[col].to_numpy() * 3.086e16  # pc to meters
                fmin_vals = df['fmin'].to_numpy()
                df[f'logEIRPmin_{key}'] = log_4pi + 2 * np.log10(d_m) + np.log10(fmin_vals) - np.log10(df['scaling_factor'])
            return df
        def analyze_shells_with_uncertainty_cumulative_log_gaia(df, log_eirp_shells):
            results = []
            nu_rel = df['nu_rel'].mean()
            fmin_shell = df['fmin'].min()
            for log_shell_val in log_eirp_shells:
                shell_val = 10**log_shell_val
                n_stars = np.sum(df['logEIRPmin_med'] <= log_shell_val)
                pos_err = np.sum((df['logEIRPmin_med'] > log_shell_val) & (df['logEIRPmin_lo'] <= log_shell_val))
                neg_err = np.sum((df['logEIRPmin_med'] <= log_shell_val) & (df['logEIRPmin_hi'] > log_shell_val))
                if n_stars == 0:
                    pos_err = 1
                    neg_err = 0
                n_stars_lower = n_stars - neg_err if (n_stars - neg_err) > 0 else np.nan
                n_stars_upper = n_stars + pos_err if (n_stars + pos_err) > 0 else np.nan
                cwtfm = (shell_val / 1e13) * (0.5 / nu_rel) * (1000 / n_stars) if n_stars > 0 else np.nan
                cwtfm_pos = (shell_val / 1e13) * (0.5 / nu_rel) * (1000 / n_stars_lower) if n_stars_lower > 0 else np.nan
                cwtfm_pos_err = cwtfm_pos - cwtfm if n_stars_lower > 0 else np.nan
                cwtfm_neg = (shell_val / 1e13) * (0.5 / nu_rel) * (1000 / n_stars_upper) if n_stars_upper > 0 else np.nan
                cwtfm_neg_err = cwtfm - cwtfm_neg if n_stars_upper > 0 else np.nan
                def safe_log10(x): return np.log10(x) if (x is not None and x > 0) else np.nan
                logTR = safe_log10(1/(n_stars * nu_rel)) if n_stars > 0 else np.nan
                logTR_pos = safe_log10(1/(n_stars_lower * nu_rel)) if n_stars_lower > 0 else np.nan
                logTR_neg = safe_log10(1/(n_stars_upper * nu_rel)) if n_stars_upper > 0 else np.nan
                logTR_pos_err = logTR_pos - logTR if (not np.isnan(logTR_pos) and not np.isnan(logTR)) else np.nan
                logTR_neg_err = logTR - logTR_neg if (not np.isnan(logTR) and not np.isnan(logTR_neg)) else np.nan
                max_dist_m = np.sqrt(shell_val / (4 * np.pi * fmin_shell))
                max_dist_pc = max_dist_m / 3.086e16
                results.append({
                    'log_EIRPmin_shell': log_shell_val,
                    'n_stars': n_stars,
                    'n_stars_pos_err': pos_err,
                    'n_stars_neg_err': neg_err,
                    'log_TR': logTR,
                    'log_TR_pos_err': logTR_pos_err,
                    'log_TR_neg_err': logTR_neg_err,
                    'CWTFM': cwtfm,
                    'CWTFM_pos_err': cwtfm_pos_err,
                    'CWTFM_neg_err': cwtfm_neg_err,
                    'max_distance_pc': max_dist_pc
                })
            return pd.DataFrame(results)

        if log_eirp_shells_local is None:
            d_max_m = limit_distance_pc * 3.086e16
            fmin_max = df_band['fmin'].max()
            fmin_min = df_band['fmin'].min()
            eirpmin_max = 4 * np.pi * d_max_m**2 * fmin_max
            eirpmin_min = 4 * np.pi * (1*3.086e16)**2 * fmin_min
            log_min = np.floor(np.log10(eirpmin_min))
            log_max = np.ceil(np.log10(eirpmin_max))
            shells = n_shells_local if n_shells_local is not None else 8
            log_eirp_shells_local = np.linspace(log_min, log_max, shells)
        else:
            log_eirp_shells_local = log_eirp_shells_local

        all_df = calculate_eirpmin_log(all_df)
        shell_results = analyze_shells_with_uncertainty_cumulative_log_gaia(all_df, log_eirp_shells_local)
        if output_prefix is not None:
                all_df.to_csv(f"{output_prefix}_gaia_df.csv", index=False)
                shell_results.to_csv(f"{output_prefix}_gaia_shell_results.csv", index=False)
        return all_df, shell_results

    # Main branch: per-band splitting or all combined
    if split_by_band:
        all_df_dict = {}
        shell_results_dict = {}
        for band_name in dataframe['receiving_band'].unique():
            band_df = dataframe[dataframe['receiving_band'] == band_name].reset_index(drop=True)
            out_prefix = f"{output_prefix}_{band_name}" if output_prefix else None
            all_df, shell_results = run_pipeline(
                band_df, band_name, out_prefix,
                log_eirp_shells_outer=log_eirp_shells, n_shells_outer=n_shells)
            all_df_dict[band_name] = all_df
            shell_results_dict[band_name] = shell_results
        return all_df_dict, shell_results_dict
    else:
        out_prefix = output_prefix
        all_df, shell_results = run_pipeline(
            dataframe, "all", out_prefix,
            log_eirp_shells_outer=log_eirp_shells, n_shells_outer=n_shells)
        return all_df, shell_results
