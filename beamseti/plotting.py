import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import itertools

def plot_logTR_vs_logEIRPmin(
    catalog_nested_dict,
    compare_mode='per_band_across_approaches',
    bands_to_compare=None,
    approaches_to_compare=None,
    overlay=True,
    colors=None,
    markers=None,
    show=True,
    return_slopes=False
):
    if approaches_to_compare is None:
        approaches_to_compare = list(catalog_nested_dict.keys())
    all_keys = set()
    for app in approaches_to_compare:
        for band in catalog_nested_dict[app].keys():
            all_keys.add((app, band))
    if bands_to_compare is not None:
        all_keys = set([k for k in all_keys if k[1] in bands_to_compare])

    if colors is None:
        color_seq = plt.colormaps["tab10"].colors
        color_iter = itertools.cycle(color_seq)
        colors = {key: next(color_iter) for key in all_keys}
    if markers is None:
        base_markers = ['o', 's', 'd', 'X', '*', 'P', 'x']
        marker_iter = itertools.cycle(base_markers)
        markers = {key: next(marker_iter) for key in all_keys}

    if overlay:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        figs, axs = [], []

    slopes_dict = {}

    vlines = [(13, 'darkblue', 'Arecibo Radar'), (16, 'magenta', 'Kardashev I'),
              (26, 'orange', 'Kardashev II')]

    def get_eirp_col(approach):
        return 'log_EIRPmin_shell' if approach.lower() in ['gaia','synthpop', 'ned'] else 'log_EIRPmin'
        
    def get_error_bars(df, approach):
        # For SynthPop and NED: symmetric, for Gaia/Uno: asymmetric
        if approach.lower() in ['gaia', 'uno']:
            if 'log_TR_pos_err' in df.columns and 'log_TR_neg_err' in df.columns:
                yerr_lower = df['log_TR_neg_err'].to_numpy()
                yerr_upper = df['log_TR_pos_err'].to_numpy()
                return np.array([yerr_lower, yerr_upper])
            else:
                return None
        else:
            if 'log_TR_err' in df.columns:
                return df['log_TR_err'].to_numpy()
            else:
                return None

    def plot_points(app, band, df, ax_):
        eirp_col = get_eirp_col(app)
        x = df[eirp_col].to_numpy()
        y = df['log_TR'].to_numpy()
        yerr = get_error_bars(df, app)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if yerr is not None:
            if isinstance(yerr, np.ndarray) and yerr.ndim == 2:
                yerr = yerr[:, mask]
            else:
                yerr = yerr[mask]
        key = (app, band)
        mcol = colors[key]
        mmkr = markers[key]
        if yerr is not None:
            ax_.errorbar(x, y, yerr=yerr, fmt=mmkr, color=mcol, alpha=0.8, markersize=6, capsize=3,
                         label=f"{app} {band}")
        else:
            ax_.scatter(x, y, color=mcol, marker=mmkr, s=45, alpha=0.80, label=f"{app} {band}")
        return x, y

    # Per-band across-approaches
    if compare_mode == 'per_band_across_approaches':
        if not bands_to_compare:
            bands_to_compare = list(next(iter(catalog_nested_dict.values())).keys())
        for band in bands_to_compare:
            all_x, all_y = [], []
            for app in approaches_to_compare:
                if band not in catalog_nested_dict[app]: continue
                df = catalog_nested_dict[app][band]
                if overlay:
                    x, y = plot_points(app, band, df, ax)
                else:
                    fig, ax_ = plt.subplots(figsize=(8, 6))
                    x, y = plot_points(app, band, df, ax_)
                    figs.append(fig)
                    axs.append(ax_)
                all_x.extend(x)
                all_y.extend(y)
                # Per-line fit
                if len(x) > 1:
                    slope, intercept, *_ = linregress(x, y)
                    slopes_dict[(app, band)] = slope
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 80)
                    yfit = slope * xfit + intercept
                    # if overlay:
                    #     ax.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7,
                    #             label=f'{app} {band} fit (slope={slope:.2f})')
                    # else:
                    #     ax_.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7,
                    #              label=f'{app} {band} fit (slope={slope:.2f})')
            # All together fit
            all_x = np.asarray(all_x)
            all_y = np.asarray(all_y)
            mask = np.isfinite(all_x) & np.isfinite(all_y)
            if mask.sum() > 1 and overlay:
                slope, intercept, *_ = linregress(all_x[mask], all_y[mask])
                xfit = np.linspace(np.nanmin(all_x[mask]), np.nanmax(all_x[mask]), 150)
                yfit = slope * xfit + intercept
                ax.plot(xfit, yfit, 'k--', lw=1.5, alpha=0.9,
                        label=f'Band {band} All fit (slope={slope:.2f})')
                slopes_dict[f"{band}_All"] = slope

    # Per-approach across-bands
    elif compare_mode == 'per_approach_across_bands':
        for app in approaches_to_compare:
            if bands_to_compare is None:
                bands_this = list(catalog_nested_dict[app].keys())
            else:
                bands_this = [b for b in bands_to_compare if b in catalog_nested_dict[app]]
            all_x, all_y = [], []
            for band in bands_this:
                df = catalog_nested_dict[app][band]
                if overlay:
                    x, y = plot_points(app, band, df, ax)
                else:
                    fig, ax_ = plt.subplots(figsize=(8, 6))
                    x, y = plot_points(app, band, df, ax_)
                    figs.append(fig)
                    axs.append(ax_)
                all_x.extend(x)
                all_y.extend(y)
                # Per-line fit
                if len(x) > 1:
                    slope, intercept, *_ = linregress(x, y)
                    slopes_dict[(app, band)] = slope
                    xfit = np.linspace(np.nanmin(x), np.nanmax(x), 80)
                    yfit = slope * xfit + intercept
                    # if overlay:
                    #     ax.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7,
                    #             label=f'{app} {band} fit (slope={slope:.2f})')
                    # else:
                    #     ax_.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7,
                    #              label=f'{app} {band} fit (slope={slope:.2f})')
            all_x = np.asarray(all_x)
            all_y = np.asarray(all_y)
            mask = np.isfinite(all_x) & np.isfinite(all_y)
            if mask.sum() > 1 and overlay:
                slope, intercept, *_ = linregress(all_x[mask], all_y[mask])
                xfit = np.linspace(np.nanmin(all_x[mask]), np.nanmax(all_x[mask]), 150)
                yfit = slope * xfit + intercept
                ax.plot(xfit, yfit, 'k--', lw=1.5, alpha=0.9,
                        label=f'{app} All bands fit (slope={slope:.2f})')
                slopes_dict[f"{app}_All"] = slope

    # Combined across-approaches
    elif compare_mode == 'combined_across_approaches':
        all_x, all_y = [], []
        for app in approaches_to_compare:
            df = catalog_nested_dict[app]['all']
            if overlay:
                x, y = plot_points(app, 'all', df, ax)
            else:
                fig, ax_ = plt.subplots(figsize=(8, 6))
                x, y = plot_points(app, 'all', df, ax_)
                figs.append(fig)
                axs.append(ax_)
            all_x.extend(x)
            all_y.extend(y)
            if len(x) > 1:
                slope, intercept, *_ = linregress(x, y)
                slopes_dict[app] = slope
                xfit = np.linspace(np.nanmin(x), np.nanmax(x), 90)
                yfit = slope * xfit + intercept
                # if overlay:
                #     ax.plot(xfit, yfit, color=colors.get((app, 'all'), f'C{approaches_to_compare.index(app)}'), 
                #             lw=2, alpha=0.7, label=f'{app} fit (slope={slope:.2f})')
                # else:
                #     ax_.plot(xfit, yfit, color=colors.get((app, 'all'), f'C{approaches_to_compare.index(app)}'), 
                #              lw=2, alpha=0.7, label=f'{app} fit (slope={slope:.2f})')
        all_x = np.asarray(all_x)
        all_y = np.asarray(all_y)
        mask = np.isfinite(all_x) & np.isfinite(all_y)
        if mask.sum() > 1 and overlay:
            slope, intercept, *_ = linregress(all_x[mask], all_y[mask])
            xfit = np.linspace(np.nanmin(all_x[mask]), np.nanmax(all_x[mask]), 180)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, 'k--', lw=1.7, alpha=0.98,
                    label=f'All approaches fit (slope={slope:.2f})')
            slopes_dict["All"] = slope

    # Multi-band multi-approach
    elif compare_mode == 'multi_band_multi_approach':
        all_x, all_y = [], []
        for app, band in all_keys:
            df = catalog_nested_dict[app][band]
            if overlay:
                x, y = plot_points(app, band, df, ax)
            else:
                fig, ax_ = plt.subplots(figsize=(8, 6))
                x, y = plot_points(app, band, df, ax_)
                figs.append(fig)
                axs.append(ax_)
            all_x.extend(x)
            all_y.extend(y)
            if len(x) > 1:
                slope, intercept, *_ = linregress(x, y)
                slopes_dict[(app, band)] = slope
                xfit = np.linspace(np.nanmin(x), np.nanmax(x), 80)
                yfit = slope * xfit + intercept
                # if overlay:
                #     ax.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7, label=f'{app} {band} fit (slope={slope:.2f})')
                # else:
                #     ax_.plot(xfit, yfit, color=colors[(app, band)], lw=2, alpha=0.7, label=f'{app} {band} fit (slope={slope:.2f})')
        all_x = np.asarray(all_x)
        all_y = np.asarray(all_y)
        mask = np.isfinite(all_x) & np.isfinite(all_y)
        if mask.sum() > 1 and overlay:
            slope, intercept, *_ = linregress(all_x[mask], all_y[mask])
            xfit = np.linspace(np.nanmin(all_x[mask]), np.nanmax(all_x[mask]), 150)
            yfit = slope * xfit + intercept
            ax.plot(xfit, yfit, 'k--', lw=1.5, alpha=0.9,
                    label=f'All bands & approaches fit (slope={slope:.2f})')
            slopes_dict['All'] = slope

    # Draw vertical/benchmark lines
    if overlay:
        for xpos, vcolor, label in vlines:
            ax.axvline(x=xpos, ls='--', color=vcolor, lw=1.2, alpha=0.55)
            ax.text(xpos+0.10, ax.get_ylim()[0]+0.1, label, color=vcolor, rotation=90, va='bottom', ha='left', fontsize=12, fontweight='bold', alpha=0.65)
        ax.set_xlabel(r'$\log_{10}\ \mathrm{EIRP_{min}}\ \mathrm{(log(W))}$')
        ax.set_ylabel(r'$\log_{10}\ T\!R$')
        ax.legend(fontsize=10)
        ax.set_title("log(TR) vs. log(EIRP$_{min}$)")
        ax.grid(True, alpha=0.22)
        plt.tight_layout()
        if show: plt.show()

    if not overlay:
        for ax_ in axs:
            ax_.set_xlabel(r'$\log_{10}\ \mathrm{EIRP_{min}}\ \mathrm{(log(W))}$')
            ax_.set_ylabel(r'$\log_{10}\ T\!R$')
            ax_.legend(fontsize=10)
            ax_.set_title("log(TR) vs. log(EIRP$_{min}$)")
            ax_.grid(True, alpha=0.22)
            plt.tight_layout()
            if show: ax_.figure.show()

    if return_slopes:
        if overlay:
            return (fig, ax, slopes_dict)
        else:
            return (figs, axs, slopes_dict)
    else:
        if overlay:
            return (fig, ax)
        else:
            return (figs, axs)






import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_density_colored_cmd_catalog_compare(
    catalog_dict,
    bands_to_compare=None,
    overlay=True,
    cmap_names=None,
    point_size=8,
    xlim=None, ylim=None,
    show=True,
    title=None
):
    """
    Plot single or overlaid density-colored CMDs for Gaia & SynthPop, per or across any band names.
    Gaia points are always plotted on top for visibility.
    Each (approach, band) pair gets a distinct fixed colormap and independent color normalization.
    Separate colorbars are created per plot.
    If comparing within the sample approach across bands (Gaia GBT-L and GBT-S) you may want to set overlay=False as there will
    probably be lots of overlap 

    Parameters
    ----------
    catalog_dict : dict
        Dict: {'Gaia': {'band1': df1, ...}, 'SynthPop': {'band2': df2, ...}}
        Each inner dict contains DataFrames per band.
    bands_to_compare : list or None
        List of band names to include. If None, plots all bands available.
    overlay : bool
        If True, overlays all selected CMDs on one plot with separate colorbars.
        If False, creates separate plot per approach-band.
    cmap_names : dict or None
        Dict mapping (approach, band) to colormap name or Colormap.
        Example: {('Gaia', 'band1'): 'viridis', ('SynthPop', 'band1'): 'plasma',
        ('Gaia', 'band2'): 'cividis', ('SynthPop', 'band2'): 'inferno'}
        If None, uses standard choices.
        If None, automatically assigned distinct colormaps.
    point_size : int, default 8
        Scatter point size.
    xlim, ylim : optional
        Axis limits as tuples.
    show : bool
        Whether to display plots immediately.
    title : str or None
        Plot title for overlay; ignored for separate plots.

    Returns
    -------
    fig, ax if overlay else (list of figs, list of axs)
    """

    # Collect all (app, band) pairs from the catalog
    all_pairs = []
    for app, banddict in catalog_dict.items():
        for band in banddict:
            all_pairs.append((app, band))

    # Filter pairs by bands_to_compare if provided
    if bands_to_compare is not None:
        all_pairs = [(app, band) for app, band in all_pairs if band in bands_to_compare]

    # Assign default auto colormaps if cmap_names not given
    auto_cmaps = [
        'viridis', 'cool', 'inferno', 'winter', 'plasma', 'cividis', 'Wistia', 'magma'
    ]
    if cmap_names is None:
        cmap_names = {}
        for i, ab in enumerate(all_pairs):
            cmap_names[ab] = auto_cmaps[i % len(auto_cmaps)]

    # Prepare entries with data and plotting info
    entries = []
    for app, band in all_pairs:
        df = catalog_dict[app][band]
        if app.lower() == "gaia":
            if not all(col in df.columns for col in ['bp_rp','abs_g_photogeo']):
                continue
            x = df['bp_rp']
            y = df['abs_g_photogeo']
            xlab, ylab = 'BP - RP (mag)', 'Abs G-band Mag (photogeo)'
        else:
            if not all(c in df.columns for c in ['Gaia_BP_EDR3', 'Gaia_RP_EDR3', 'Gaia_G_EDR3']):
                continue
            x = df['Gaia_BP_EDR3'] - df['Gaia_RP_EDR3']
            y = df['Gaia_G_EDR3']
            xlab, ylab = 'BP − RP (mag)', 'Abs G-band Mag (SynthPop)'
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            continue
        entries.append({
            'app': app,
            'band': band,
            'x': x[mask].copy(),
            'y': y[mask].copy(),
            'xlab': xlab,
            'ylab': ylab,
            'cmap': plt.get_cmap(cmap_names[(app, band)]),
            'label': f"{app} {band}"
        })

    # Sort entries so Gaia plotted on top
    entries_sorted = [e for e in entries if e['app'].lower() != 'gaia'] + [e for e in entries if e['app'].lower() == 'gaia']

    if overlay:
        fig, ax = plt.subplots(figsize=(7, 9))
        colorbars = []
        xlabels_seen = set(e['xlab'] for e in entries_sorted)
        ylabels_seen = set(e['ylab'] for e in entries_sorted)

        for e in entries_sorted:
            xy = np.vstack([e['x'], e['y']])
            try:
                density = gaussian_kde(xy)(xy)
            except Exception:
                density = np.ones_like(e['x'])

            sqrt_density = np.sqrt(density)
            norm = mcolors.Normalize(vmin=sqrt_density.min(), vmax=sqrt_density.max())

            sc = ax.scatter(
                e['x'], e['y'], c=sqrt_density,
                cmap=e['cmap'], norm=norm,
                s=point_size,
                alpha=1 if e['app'].lower() == 'gaia' else 1,
                edgecolor='none',
                label=e['label'],
                zorder=2 if e['app'].lower() == 'gaia' else 1,
            )
            colorbars.append({'handle': sc, 'label': e['label']})

        ax.invert_yaxis()
        ax.set_xlabel(xlabels_seen.pop() if len(xlabels_seen) == 1 else 'Color (mag)')
        ax.set_ylabel(ylabels_seen.pop() if len(ylabels_seen) == 1 else 'Absolute Magnitude (mag)')
        ax.set_title(title or "CMD Comparison: " + ", ".join(e['label'] for e in entries_sorted))
        ax.legend(fontsize=10)
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        ax.grid(True, alpha=0.15)
        plt.tight_layout()

        # Add individual colorbars per scatter plot
        divider = make_axes_locatable(ax)
        pad = 0.6
        for i, cb in enumerate(colorbars):
            cax = divider.append_axes("right", size="3.5%", pad=pad + i * 0.07)
            cb_obj = plt.colorbar(cb['handle'], cax=cax)
            cb_obj.set_label('Square Root Density', rotation=270, labelpad=15, fontsize=9)
            cb_obj.ax.tick_params(labelsize=12)

        if show:
            plt.show()
        return fig, ax

    else:
        figs, axs = [], []
        for e in entries_sorted:
            fig, ax = plt.subplots(figsize=(7, 9))
            xy = np.vstack([e['x'], e['y']])
            try:
                density = gaussian_kde(xy)(xy)
            except Exception:
                density = np.ones_like(e['x'])

            sqrt_density = np.sqrt(density)
            norm = mcolors.Normalize(vmin=sqrt_density.min(), vmax=sqrt_density.max())

            sc = ax.scatter(
                e['x'], e['y'], c=sqrt_density,
                cmap=e['cmap'], norm=norm, s=point_size,
                alpha=1 if e['app'].lower() == 'gaia' else 1,
                edgecolor='none', label=e['label']
            )
            ax.invert_yaxis()
            ax.set_xlabel(e['xlab'])
            ax.set_ylabel(e['ylab'])
            ax.set_title(title or f"{e['label']} CMD")
            if xlim: ax.set_xlim(xlim)
            if ylim: ax.set_ylim(ylim)
            ax.grid(True, alpha=0.15)
            plt.tight_layout()
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label('Square Root Density', rotation=270, labelpad=17, fontsize=12)
            cb.ax.tick_params(labelsize=12)
            figs.append(fig)
            axs.append(ax)
        if show:
            for fig in figs:
                plt.show()
        return figs, axs
