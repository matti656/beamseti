Section: Using Redshift-Independent Distances
If you want to include NED’s redshift-independent distances in your analysis, you’ll need to use the provided `ned_redshift_independent_distances.csv` file (included in this repo) and the helper function `get_ned_redshift_independent_distances`:
from yourpackage import get_ned_redshift_independent_distances, process_ned_catalog

ned_data = get_ned_redshift_independent_distances()  # Loads from Github by default
ned_df, constraints_df, field_summary, region_summary = process_ned_catalog(
    ned_df=my_ned_df,
    use_ri_distances=True,
    ned_data=ned_data,
    ...
)
