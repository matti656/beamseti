import numpy as np
import pandas as pd
from s2sphere import LatLng, Cap, RegionCoverer, CellId, CellUnion, Angle, Cell

def process_uno(
    dataframe,
    log_rho_dict={'lower': 8.443, 'nominal': 8.473, 'upper': 8.653},
    log_rho_nominal=None, log_rho_pos_err=None, log_rho_neg_err=None,
    split_by_band=False,
    output_prefix=None,
    verbose=True
):
    """
    Extragalactic SETI pipeline following the GSMF-based methodology of Uno et al. (2023)
    for constraining transmitter rates in wide-field surveys via field overlap, galaxy stellar mass function (GSMF) statistics,
    and controlled systematics using a range of initial mass functions (IMFs).
    
    This function processes a DataFrame of sky fields, computes the overlap-corrected unique area using S2 geometry,
    converts the observed volume to total stellar mass using GSMF (Baldry et al. 2012; Driver et al. 2022 constants),
    propagates systematic uncertainties per IMF as in Uno et al. (2023), and returns population and EIRPmin constraints
    for all fields together or per receiving band.
    
    Parameters
    ----------
    dataframe : pandas.DataFrame
        Required columns and types:
          - 'ra'            [degrees]: ICRS Right Ascension of field center, converted to float internally.
          - 'dec'           [degrees]: ICRS Declination of field center, converted to float internally.
          - 'fwhm_arcmin'   [arcmin]: Full Width at Half Maximum of field/beam, converted to float internally.
          - 'dlim_Mpc'      [Mpc]: Maximum distance limit for all fields in a band, converted to float internally.
          - 'nu_rel'        [unitless]: total bandwidth of the receiver normalized by the central, converted internally to float
            observing frequency (used in Transmitter Rate and CWTFM scaling)
          - 'fmin'          [W * m^-2]: Minimum detectable flux density.
          - 'field name': Unique field identifier, converted internally to str
          - 'receiving_band': Name of the observing band (e.g., 'L', 'S'); allows per-band separation, converted internally to str
        
        All columns must be convertible to their specified types.
    
    log_rho_dict : dict, optional
        Default: {'lower': 8.443, 'nominal': 8.473, 'upper': 8.653}
        log₁₀(stellar mass density [M⊙/Mpc³]) for systematic GSMF evaluation:
          - lower: Chabrier IMF (Chabrier 2003)
          - nominal: Kroupa IMF (Kroupa 2001)
          - upper: Salpeter IMF (Salpeter 1955)
        (See Uno et al. 2023; systematic offset only; statistical error ≪ systematics)
    
    log_rho_nominal, log_rho_pos_err, log_rho_neg_err : float, optional
        If provided, override log_rho_dict with direct nominal log₁₀ stellar mass density and positive/negative error.
    
    split_by_band : bool, optional
        If True, calculates summary statistics per unique receiving_band (returns dict with one row per band).
        If False, combines all fields (across all bands) into a single area/volume calculation.
    
    output_prefix : str, optional
        If provided, concatenated results are saved in CSV files using this prefix (per band, or combined)
        summary_df is saved according to output_prefix_band name_uno.csv      
    
    verbose : bool, optional
        If True, prints processing progress and overlap/S2 warnings.
    
    Returns
    -------
    If split_by_band == False:
        summary_df : pandas.DataFrame
            One-row DataFrame with total area, volume, stellar mass (systematics-propagated), number of stars,
            log(TR) and its errors, CWTFM and its errors, and input parameters.
    
    If split_by_band==True: {band: summary_df} dict (one row per band)
            Returns per-receiving_band DataFrames as dictionary entries.
            access each DataFrame by summary_dict['band 1']...,
    
    Methodology and Equations (Uno et al. 2023; GSMF)
    -------------------------------------------------
    - Stellar mass density: Computes log₁₀(M⋆/M⊙) using the GSMF from Baldry et al. (2012).
        Default systematic envelope ('lower', 'nominal', 'upper') corresponds to Chabrier/Kroupa/Salpeter IMFs
        (see Table 1, Uno et al. 2023). Values for log₁₀ρ and constants from Driver et al. (2022) are also supported.
    - Volume calculation: Unique area for all beams (overlap-subtracted) is calculated using s2sphere for multi-field mosaics,
        or analytic πr² for single fields. Total search volume:
            Ω = unique area [deg²]
            Vr = (Ω_sky [sr] / 3) × dlim_Mpc³
    - Stellar mass and stars: Total stellar mass M_gal = ρ × Vr, uncertainties from GSMF systematics.
        Number of stars N_stars = M_gal (assuming 1 solar mass per star, Uno et al. 2023, §2).
        N_stars_err propagates uncertainty in log(Mgal/M⊙) from different (lower, nominal, upper) IMF log ρ values.
    - Transmitter Rate (TR): log_TR = -log₁₀(N_stars) - log₁₀(nu_rel), with errors from IMF range.
    - CWTFM (Figure of Merit): CWTFM = (500 / 1e13) × EIRPmin / (N_stars × nu_rel), as in modern SETI literature.
        All quantities are calculated for the entire overlap-corrected region.
    - EIRPmin: Calculated at maximal survey distance for the field, using
          log₁₀(EIRPmin) = log₁₀(4π) + 2 log₁₀(dlim_Mpc × 1e6 × 3.086e16 [m]) + log₁₀(fmin)
          Assumes a beam scaling of 1 across all fields.
    - Systematic errors: Systematic uncertainties are evaluated using the spread between IMF choices,
        following Uno et al. (2023): nominal (Kroupa), lower (Chabrier), upper (Salpeter).
    
    Units
    -----
    - RA, Dec: [degrees]
    - fwhm_arcmin: [arcminutes]
    - dlim_Mpc: [Mpc]
    - fmin: [W * m^-2]
    - nu_rel: nu_rel(BW/f): unitless, where f represents the observed frequency, 
        and BW denotes the total bandwidth in GHz
    - area_deg2: [deg²]
    - Vr_Mpc3: [Mpc³]
    - log_Mgal: log₁₀(M⋆ [M⊙])
    - N_stars: total number of stars, assuming 1M⊙ per star.
    - log_TR: log₁₀(Transmitter Rate constraint)
    - log_EIRPmin: log₁₀(W).
    - CWTFM: unitless (Figure of Merit)
    
    Field geometry and overlap correction:
    --------------------------------------
    - Unique (non-overlapping) area is calculated using s2sphere S2 geometry.
    - For a single field, an analytic πr² estimate is used (Uno et al. 2023, §2).
    - For multiple fields, full S2 union is computed.
    - This ensures population and EIRPmin calculation is only for the true non-duplicated search volume.
    
    split_by_band logic:
    --------------------
    - If split_by_band == True: Computes all statistics per unique 'receiving_band' in the DataFrame.
        All fields for a given band are combined before overlap correction and statistics.
    - If split_by_band == False: Returns one set of statistics for all fields across all bands combined.
    
    References
    ----------
    - Uno, Y., Hashimoto, T., Goto, T., et al. (2023), "Upper limits on transmitter rate of extragalactic civilizations
        placed by Breakthrough Listen observations," MNRAS, 522, 4649. https://doi.org/10.1093/mnras/stad993
    - Baldry, I. K., et al. (2012), "Galaxy And Mass Assembly (GAMA): the galaxy stellar mass function at z < 0.06," MNRAS, 421, 621.
    - Driver, S. P., et al. (2022), "The Galaxy Stellar Mass Function: From the Kilo-Degree Survey," MNRAS, 510, 1056.
    - Chabrier, G. (2003), Kroupa, P. (2001), Salpeter, E. E. (1955) — for IMF systematics.
    """
    dataframe = dataframe.copy()
    dataframe.loc[:, 'field name'] = dataframe.loc[:, 'field name'].astype(str)
    dataframe.loc[:, 'receiving_band'] = dataframe.loc[:, 'receiving_band'].astype(str)
    dataframe.loc[:, 'ra'] = dataframe.loc[:, 'ra'].astype(float)
    dataframe.loc[:, 'dec'] = dataframe.loc[:, 'dec'].astype(float)
    dataframe.loc[:, 'fwhm_arcmin'] = dataframe.loc[:, 'fwhm_arcmin'].astype(float)
    dataframe.loc[:, 'dlim_Mpc'] = dataframe.loc[:, 'dlim_Mpc'].astype(float)
    dataframe.loc[:, 'fmin'] = dataframe.loc[:, 'fmin'].astype(float)
    dataframe.loc[:, 'nu_rel'] = dataframe.loc[:, 'nu_rel'].astype(float)

    def fwhm_fields_to_area_deg2_s2(ra_arr, dec_arr, fwhm_arr, max_cells=10000):
        if len(ra_arr) == 1:
            radius_deg = (fwhm_arr[0]/2) / 60.0
            area_deg2 = np.pi * radius_deg**2
            # if verbose:
            #     print("S2 union unnecessary for single field; using analytic area.")
            return area_deg2
        cell_union = set()
        coverer = RegionCoverer()
        coverer.max_cells = max_cells
        for ra, dec, fwhm in zip(ra_arr, dec_arr, fwhm_arr):
            radius_arcmin = fwhm / 2
            radius_rad = np.deg2rad(radius_arcmin / 60.0)
            center = LatLng.from_degrees(dec, ra).to_point()
            cap = Cap.from_axis_angle(center, Angle.from_radians(float(radius_rad)))
            cells = coverer.get_covering(cap)
            for cell in cells:
                cell_union.add(cell.id())
        cell_union = CellUnion(list(cell_union))
        area_sr = sum(Cell(i).get_rect_bound().area() for i in cell_union.cell_ids())
        area_deg2 = area_sr * (180.0/np.pi)**2
        return area_deg2

    def uno_summary_from_df(df, save_path=None):
        ra_arr = df['ra'].astype(float).to_numpy()
        dec_arr = df['dec'].astype(float).to_numpy()
        fwhm_arr = df['fwhm_arcmin'].astype(float).to_numpy()
        dlim_Mpc = df['dlim_Mpc'].astype(float).iloc[0]
        nu_rel = df['nu_rel'].astype(float).iloc[0]
        fmin = df['fmin'].astype(float).iloc[0]
        total_area_deg2 = fwhm_fields_to_area_deg2_s2(ra_arr, dec_arr, fwhm_arr)
        Omega_tot = total_area_deg2 * (np.pi/180)**2
        k_tot = Omega_tot / 3
        Vr_tot = k_tot * dlim_Mpc**3

        use_nominal_mode = (log_rho_nominal is not None)
        if use_nominal_mode:
            log_Mgal_nom = log_rho_nominal + np.log10(Vr_tot)
            pos_err = log_rho_pos_err if log_rho_pos_err is not None else 0
            neg_err = log_rho_neg_err if log_rho_neg_err is not None else 0
            log_Mgal_upper = log_Mgal_nom + pos_err
            log_Mgal_lower = log_Mgal_nom - neg_err
        else:
            log_Mgal = {IMF: log_rho + np.log10(Vr_tot) for IMF, log_rho in log_rho_dict.items()}
            log_Mgal_nom = log_Mgal['nominal']
            log_Mgal_upper = log_Mgal['upper']
            log_Mgal_lower = log_Mgal['lower']
            pos_err = log_Mgal_upper - log_Mgal_nom
            neg_err = log_Mgal_nom - log_Mgal_lower
        N_stars_nom = 10**log_Mgal_nom
        N_stars_upper = 10**log_Mgal_upper
        N_stars_lower = 10**log_Mgal_lower
        log_TR = -log_Mgal_nom - np.log10(nu_rel)
        log_TR_pos_err = neg_err
        log_TR_neg_err = pos_err
        d_m = dlim_Mpc * 1e6 * 3.086e16
        log_4pi = np.log10(4 * np.pi)
        log_EIRPmin = log_4pi + 2 * np.log10(d_m) + np.log10(fmin)
        EIRPmin = 10**log_EIRPmin
        CWTFM_nom = (500 / 1e13) * EIRPmin / (N_stars_nom * nu_rel)
        CWTFM_upper = (500 / 1e13) * EIRPmin / (N_stars_upper * nu_rel)
        CWTFM_lower = (500 / 1e13) * EIRPmin / (N_stars_lower * nu_rel)
        CWTFM_pos_err = CWTFM_lower - CWTFM_nom
        CWTFM_neg_err = CWTFM_nom - CWTFM_upper
        row = {
            'area_deg2': total_area_deg2,
            'dlim_Mpc': dlim_Mpc,
            'Vr_Mpc3': Vr_tot,
            "nu_rel": nu_rel,
            'fmin': fmin,
            'log_Mgal': log_Mgal_nom,
            'log_Mgal_pos_err': pos_err,
            'log_Mgal_neg_err': neg_err,
            'N_stars': N_stars_nom,
            'N_stars_upper': N_stars_upper,
            'N_stars_lower': N_stars_lower,
            'log_TR': log_TR,
            'log_TR_pos_err': log_TR_pos_err,
            'log_TR_neg_err': log_TR_neg_err,
            'log_EIRPmin': log_EIRPmin,
            'CWTFM': CWTFM_nom,
            'CWTFM_pos_err': CWTFM_pos_err,
            'CWTFM_neg_err': CWTFM_neg_err
        }
        summary_df = pd.DataFrame([row])
        if save_path is not None:
            summary_df.to_csv(save_path, index=False)
        return summary_df

    if split_by_band:
        summary_dict = {}
        for band, band_df in dataframe.groupby('receiving_band'):
            if verbose:
                print(f"Processing receiving_band={band} with {len(band_df)} fields.")
            save_path = f"{output_prefix}_{band}_uno.csv" if output_prefix else None
            summary_dict[band] = uno_summary_from_df(band_df, save_path=save_path)
        return summary_dict
    else:
        if verbose:
            print(f"Processing all bands together ({len(dataframe)} fields).")
        save_path = f"{output_prefix}_uno.csv" if output_prefix else None
        return uno_summary_from_df(dataframe, save_path=save_path)
