import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import synthpop
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_synthpop(
    dataframe,
    limit_distance_pc=10000,
    output_prefix=None,
    batch_size=10,
    max_workers=5,
    sleep_between_batches=5,
    n_shells=8,
    log_eirp_shells=None,
    config_file='huston2025_defaults.synthpop_conf',
    split_by_band=False
):
    """
    Parallelized, per-field SynthPop stellar catalog simulation and SETI sensitivity analysis,
    adapting the methodology of Wlodarczyk-Sroka et al. (2020, "WG&S 2020"), Bailer-Jones et al. (2021),
    and using the synthetic Milky Way population model as described in Klüter et al. (2025, "SynthPop").
    Although this function is written in a batch-compatible form for pipeline consistency,
    the SynthPop engine only allows one field to be processed at a time; thus, batch_size and max_workers
    have no effect (processing is strictly serial).
    
    Each field is simulated as a synthetic sightline catalogue via SynthPop, with the returned stars assigned
    beam response factors, analyzed for minimum detectable EIRP, and then summarized in shell sensitivity tables
    using Poisson statistics for star count uncertainties.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Required columns:
        - 'ra'           [degrees]: ICRS Right Ascension (field/beam center, converted internally to float)
        - 'dec'          [degrees]: ICRS Declination (field/beam center, converted internally to float)
        - 'fwhm_arcmin'  [arcminutes]: Beamwidth (FWHM) of the field, converted internally to float
        - 'fmin'         [W * m^−2]: Minimum detectable flux density at the field (per band), converted internally to float
        - 'nu_rel'       [unitless]: total bandwidth of the receiver normalized by the central
            observing frequency (used in Transmitter Rate and CWTFM scaling), converted internally to float|
        - 'field name': Unique field identifier, converted internally to str
        - 'receiving_band': Name of the observing band (e.g., 'L', 'S'); allows per-band separation, converted internally to str
    
        Acceptable types: all columns must be convertible to their indicated types.
    
    limit_distance_pc : float
        Maximum stellar distance [pc] for target selection (default: 10,000 pc, as in WG&S 2020).
    output_prefix : str, optional
        If provided, concatenated results are saved in CSV files using this prefix (per band, or combined)
        all_df is saved according to output_prefix_band name_synthpop_df.csv
        shell_results is saved according to output_prefix_band name_synthpop_shell_results.csv    
    batch_size, max_workers, sleep_between_batches : [ignored]
        Included for API compatibility—SynthPop can only process one field at a time, so these are ignored.
    n_shells : int, optional
        Number of log₁₀(EIRPmin) sensitivity shells (default: 8, per WG&S 2020).
    log_eirp_shells : array-like, optional
        If specified, manual edges (log₁₀ EIRPmin [W]) for sensitivity shell analysis. Per WG&S 2020, for max distances less 
        than 10 kpc the standard range of EIRPmin shells is np.linspace(11, 18, 8).
    config_file : str
        Name (or path) of SynthPop configuration file (see SynthPop documentation or Klüter et al. 2025).
    split_by_band : bool, optional
        If True, analysis is run per unique receiving_band and outputs are dictionaries keyed by band.
        If False (default), all input fields are combined for analysis.
    
    Returns
    -------
    If split_by_band == False:
        all_df : pandas.DataFrame
            Full synthetic catalog for all input fields, with (per star):
                - Equatorial and Galactic coordinates
                - Photometric properties (e.g., Gaia_G_EDR3, BP-RP)
                - Distance [pc], as given by synthetic model
                - Field/band info, fmin, nu_rel, fwhm_arcmin
                - θ_arcmin [arcmin]: Angular offset from beam center
                - scaling_factor_percent [%]: Fractional beam response (Gaussian, see below)
                - logEIRPmin [log₁₀(W)]: Per-star minimum detectable transmitter power after beam attenuation.
        shell_results : pandas.DataFrame
            For each EIRPmin shell (per-band if split_by_band):
                - log_EIRPmin_shell [log₁₀(W)]
                - n_stars, n_stars_err (Poisson)
                - log_TR, log_TR_err: log₁₀(1/(n_stars * nu_rel)), plus error
                - CWTFM, CWTFM_err: Continuous Waveform Transmitter Figure-of-Merit, plus error
                - max_distance_pc [pc]: Maximum distance observable for EIRPmin at given fmin
    
    If split_by_band == True:
        all_df_dict : dict
            Per-band DataFrame for each unique 'receiving_band'
        shell_results_dict : dict
            Per-band shell analysis DataFrame
        Access each DataFrame by all_df_dict['band 1']...,
            shell_results_dict['band 1']...
    
    Methodology and Equations (WG&S 2020, Klüter+2025)
    --------------------------------------------------
    - Stellar populations for each field are simulated using SynthPop v2+ (Klüter et al. 2025), with all sightline
        geometry, dust, and synthetic photometry as controlled by the config file.
    - All stars within the (RA, Dec, FWHM) field are selected out to limit_distance_pc, with field-by-field parameters
        (fmin, nu_rel, etc.) determined by input DataFrame columns (see Table 1, Klüter+ 2025).
    - Each star is assigned a beam response:
        scaling_factor_percent = exp(−4 ln(2) * (θ_arcmin / FWHM_arcmin)^2) × 100
        following the same prescription as WG&S 2020.
    - Detection threshold EIRPmin for each star:
        log₁₀(EIRPmin) = log₁₀(4π) + 2 log₁₀(distance_m) + log₁₀(fmin) − log₁₀(scaling_fraction)
        (distance in meters; fmin in W * m^-2; scaling_fraction = scaling_factor_percent / 100)
    - Sensitivity shells (bins):
        - Default: np.linspace(11, 18, 8) in log₁₀(W) for EIRPmin (WG&S 2020, Table 2).
        - Shell analysis: for each shell edge, compute the number of stars and FOM within EIRPmin ≤ shell value.
    - Uncertainty in n_stars: Uses Poisson statistics: n_stars_err = sqrt(n_stars), as the number of detections in each shell.
    - Error propagation: 
        - log_TR_err and CWTFM_err are estimated via standard Poisson error per bin.
        - Final uncertainties are derived using standard propagation formulas.
    - Maximum distance per shell is calculated via EIRPmin_shell, per receiving band's minimum fmin.
    
    Units
    -----
    - Angles: ra, dec [degrees]; fwhm_arcmin, θ_arcmin [arcmin]
    - Distances: Dist_pc [pc] (converted to meters [m] for EIRPmin)
    - fmin: W * m^-2
    - nu_rel: nu_rel(BW/f): unitless, where f represents the observed frequency, 
        and BW denotes the total bandwidth in GHz
    - scaling_factor_percent: [%] [50, 100], normalized by beam center
    - logEIRPmin: log₁₀(W)
    - log_TR, log_TR_err: log₁₀(Transmitter Rate) [dimensionless]
    - CWTFM: unitless
    - max_distance_pc: [pc]
    
    Field geometry and split_by_band
    -------------------------------
    - Each field is simulated independently, as required by SynthPop (Klüter et al. 2025);
      batch_size and max_workers are retained for API compatibility but have no effect.
    - If split_by_band is True, each receiving_band value in 'receiving_band' column of the
      input DataFrame is processed independently (per-band outputs for multi-band SETI analyses).
    - If split_by_band is False, all fields are processed together (no band distinction).
    
    References
    ----------
    - Bailer-Jones, C. A. L., et al. (2021), "Estimating Distances from Parallaxes. V. ... Gaia EDR3," AJ, 161, 147. https://doi.org/10.3847/1538-3881/abd806
    - Klüter, J., Huston, M. J., Aronica, A., et al. (2025), "SynthPop: A New Framework for Synthetic Milky Way Population Generation," 
        AJ, 169, 317. https://doi.org/10.3847/1538-3881/adcd7a
    - Wlodarczyk-Sroka, B. S., Garrett, M. A., Siemion, A. P. V. (2020), "Extending the Breakthrough Listen nearby star survey to other 
        stellar objects in the field," MNRAS, 498, 5720. https://doi.org/10.1093/mnras/staa2672
    """
    dataframe = dataframe.copy()
    dataframe.loc[:, 'field name'] = dataframe.loc[:, 'field name'].astype(str)
    dataframe.loc[:, 'receiving_band'] = dataframe.loc[:, 'receiving_band'].astype(str)
    dataframe.loc[:, 'ra'] = dataframe.loc[:, 'ra'].astype(float)
    dataframe.loc[:, 'dec'] = dataframe.loc[:, 'dec'].astype(float)
    dataframe.loc[:, 'fwhm_arcmin'] = dataframe.loc[:, 'fwhm_arcmin'].astype(float)
    dataframe.loc[:, 'fmin'] = dataframe.loc[:, 'fmin'].astype(float)
    dataframe.loc[:, 'nu_rel'] = dataframe.loc[:, 'nu_rel'].astype(float)

    # Pipeline core as an inner function
    def run_pipeline_band(
        band_df,
        band_name,
        output_prefix=None,
        config_file=config_file,
        log_eirp_shells_outer=None,
        n_shells_outer=None
    ):
        n_total = len(band_df)

        def angular_separation_arcmin(beam_ra, beam_dec, star_ras, star_decs):
            c1 = SkyCoord(beam_ra, beam_dec, unit='deg')
            c2 = SkyCoord(star_ras, star_decs, unit='deg')
            return c1.separation(c2).arcminute

        def beam_scaling_percent(beam_ra, beam_dec, star_ras, star_decs, fwhm_arcmin):
            theta_arcmin = angular_separation_arcmin(beam_ra, beam_dec, star_ras, star_decs, )
            exponent = -4 * np.log(2) * (theta_arcmin / fwhm_arcmin)**2
            scaling_factor = np.exp(exponent) * 100
            return theta_arcmin, scaling_factor

        def process_beam_vectorized(beam_ra, beam_dec, fwhm_arcmin, df):
            df = df.copy()
            star_ras = df['ra'].to_numpy()
            star_decs = df['dec'].to_numpy()
            theta, scaling = beam_scaling_percent(beam_ra, beam_dec, star_ras, star_decs, fwhm_arcmin)
            df['theta_arcmin'] = theta
            df['scaling_factor_percent'] = scaling
            return df

        def calculate_log_eirpmin(df):
            log_4pi = np.log10(4 * np.pi)
            d_m = df["Dist_pc"].to_numpy() * 3.086e16  # pc to meters
            fmin_vals = df['fmin'].to_numpy()
            scaling_frac = df['scaling_factor_percent'].to_numpy() / 100.0
            df['logEIRPmin'] = log_4pi + 2 * np.log10(d_m) + np.log10(fmin_vals) - np.log10(scaling_frac)
            return df

        def analyze_shells_cumulative_log_synth(df, log_eirp_shells_inner):
            results = []
            for log_shell_val in log_eirp_shells_inner:
                shell_val = 10**log_shell_val
                n_stars = np.sum(df['logEIRPmin'] <= log_shell_val)
                n_stars_err = np.sqrt(n_stars)
                nu_rel_shell = df['nu_rel'].mean() if n_stars > 0 else np.nan
                cwtfm = (shell_val / 1e13) * (0.5 / nu_rel_shell) * (1000 / n_stars) if n_stars > 0 else np.nan
                cwtfm_err = cwtfm * (n_stars_err / n_stars) if n_stars > 0 else np.nan
                logTR = np.log10(1/(n_stars * nu_rel_shell)) if n_stars > 0 else np.nan
                logTR_err = np.abs(logTR * (n_stars_err / n_stars)) if n_stars > 0 else np.nan
                fmin_shell = df["fmin"].min() if n_stars > 0 else np.nan
                max_dist_m = np.sqrt(shell_val / (4 * np.pi * fmin_shell)) if (n_stars > 0 and fmin_shell > 0) else np.nan
                max_dist_pc = max_dist_m / 3.086e16 if max_dist_m is not np.nan else np.nan
                results.append({
                    'log_EIRPmin_shell': log_shell_val,
                    'n_stars': n_stars,
                    'n_stars_err': n_stars_err,
                    'log_TR': logTR,
                    'log_TR_err': logTR_err,
                    'CWTFM': cwtfm,
                    'CWTFM_err': cwtfm_err,
                    'max_distance_pc': max_dist_pc,
                })
            return pd.DataFrame(results)

        # Shell grid
        if log_eirp_shells_outer is None:
            d_max_m = limit_distance_pc * 3.086e16
            fmin_min = band_df['fmin'].min()
            shells = n_shells_outer if n_shells_outer is not None else 8
            eirpmin_max = 4 * np.pi * d_max_m**2 * fmin_min
            eirpmin_min = 4 * np.pi * (1 * 3.086e16)**2 * fmin_min
            log_min = np.floor(np.log10(eirpmin_min))
            log_max = np.ceil(np.log10(eirpmin_max))
            log_eirp_shells_to_use = np.linspace(log_min, log_max, shells)
        else:
            log_eirp_shells_to_use = log_eirp_shells_outer

        # Initialize SynthPop once (per band)
        mod = synthpop.SynthPop(
            config_file,
            extinction_map_kwargs={'name':'Surot', "use_h5":True},
            chosen_bands=['Bessell_U', 'Bessell_B', 'Bessell_V', 'Bessell_R', 'Bessell_I',
                        "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"],
            obsmag = False,
            maglim=['Bessell_I', 99, "keep"],
            post_processing_kwargs=[{"name": "ProcessDarkCompactObjects", "remove": False},
                                {"name": "equatorial_coordinates"}],
            name_for_output='mod2test'
        )
        mod.init_populations()

        # Batch jobs
        batch_files = []
        def process_sightline_batch(batch_indices, batch_idx):
            batch_results = []
            for i in batch_indices:
                row = band_df.iloc[i]
                ra_deg = float(row['ra'])
                dec_deg = float(row['dec'])
                fwhm_arcmin = float(row['fwhm_arcmin'])
                field_name = str(row['field name'])
                fmin_val = float(row['fmin'])
                nu_rel_val = float(row['nu_rel'])

                coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
                l = coord.galactic.l.deg
                b = coord.galactic.b.deg
                radius_deg = (fwhm_arcmin / 2) / 60
                solid_angle_deg2 = np.pi * radius_deg**2
                cat, distr = mod.process_location(
                    l_deg=l, b_deg=b,
                    solid_angle=solid_angle_deg2,
                    solid_angle_unit='deg^2'
                )
                cat["Dist_pc"] = cat["Dist"] * 1000
                cat = process_beam_vectorized(ra_deg, dec_deg, fwhm_arcmin, cat)
                cat['field_name'] = field_name
                cat['fmin'] = fmin_val
                cat['nu_rel'] = nu_rel_val
                cat['fwhm_arcmin'] = fwhm_arcmin
                cat = cat[(cat["Dist_pc"] <= limit_distance_pc) & (cat["Dist_pc"] >= 0)].copy()
                cat = cat[cat['scaling_factor_percent'] >= 50].copy()
                cat = cat[cat['scaling_factor_percent'] <= 100].copy()
                batch_results.append(cat)
            if batch_results:
                batch_df = pd.concat(batch_results, ignore_index=True)
                # if output_prefix is not None:
                #     filename = f"{output_prefix}_{band_name}_batch{batch_idx}.csv"
                #     batch_df.to_csv(filename, index=False)
                #     return filename
                # else:
                return batch_df
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch_start in range(0, n_total, batch_size):
                batch_end = min(batch_start + batch_size, n_total)
                batch_indices = list(range(batch_start, batch_end))
                futures.append(executor.submit(process_sightline_batch, batch_indices, batch_start // batch_size))
                time.sleep(sleep_between_batches)

            batch_results = []
            for future in as_completed(futures):
                res = future.result()
                if res is not None:
                    batch_results.append(res)

        if not batch_results:
            print("No results fetched.")
            return pd.DataFrame(), pd.DataFrame()

        # If output_prefix, batch_results are filenames; else, DataFrames
        # if output_prefix is not None:
        #     dfs = [pd.read_csv(f) for f in batch_results]
        # else:
        dfs = batch_results

        all_df = pd.concat(dfs, ignore_index=True)
        all_df = calculate_log_eirpmin(all_df)
        shell_results = analyze_shells_cumulative_log_synth(all_df, log_eirp_shells_to_use)
        drop_cols = [col for col in all_df.columns if "eirp_boost" in col or "max_distance_pc_at_eirpmin" in col]
        all_df = all_df.drop(columns=drop_cols, errors="ignore")
        column_to_move = all_df.pop('field_name')
        all_df.insert(0, 'field_name', column_to_move)
        if output_prefix is not None:
            all_df.to_csv(f"{output_prefix}_synthpop_df.csv", index=False)
            shell_results.to_csv(f"{output_prefix}_synthpop_shell_results_df.csv", index=False)
        return all_df, shell_results

    # Main Logic: Split or Not by Band
    if split_by_band:
        all_df_dict = {}
        shell_results_dict = {}
        for band_name in dataframe['receiving_band'].unique():
            df_band = dataframe[dataframe['receiving_band'] == band_name].reset_index(drop=True)
            out_prefix = f"{output_prefix}_{band_name}" if output_prefix else None
            all_df, shell_results = run_pipeline_band(
                df_band, band_name, output_prefix=out_prefix,
                log_eirp_shells_outer=log_eirp_shells, n_shells_outer=n_shells
            )
            all_df_dict[band_name] = all_df
            shell_results_dict[band_name] = shell_results
        return all_df_dict, shell_results_dict
    else:
        all_df, shell_results = run_pipeline_band(
            dataframe, "all", output_prefix=output_prefix,
            log_eirp_shells_outer=log_eirp_shells, n_shells_outer=n_shells
        )
        return all_df, shell_results
