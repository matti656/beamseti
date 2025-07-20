## Quick Start (with NED Redshift-independent Distances)

1. Clone this repo
2. In your notebook/script, import:
    ```
    from yourpackage import process_ned_catalog, get_ned_redshift_independent_distances
    ```

3. Load your object catalog as `my_ned_df` (see notebook/example), then:

    ```
    ned_data = get_ned_redshift_independent_distances()
    ned_df, constraints_df, field_summary, region_summary = process_ned_catalog(
        ned_df=my_ned_df,
        use_ri_distances=True,
        ned_data=ned_data,
        ...
    )
    ```
