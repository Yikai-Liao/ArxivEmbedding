import polars as pl
from .name import *
from pathlib import Path

def order_by_category_and_date(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Orders the DataFrame first by 'category' in ascending order,
    then by 'date' in descending order within each category.
    """
    return df.sort(
        by=[
            pl.col(CATEGORIES).list.first(),
            pl.col(PUBLISH_DATE)
        ],
        descending=[False, True]
    )


def dump_metadata(df: pl.DataFrame, path: str | Path, row_group: int = 50_000):
    """
    Dumps the DataFrame into Parquet files partitioned by year.

    Parameters:
    - df: The input DataFrame containing metadata.
    - path: The directory where the Parquet files will be saved.
    - row_group: The number of rows per row group in the Parquet file.
    """
    path = Path(path)
    assert not path.exists() or path.is_file()
    
    df.write_parquet(
        path,
        row_group_size=row_group,
        compression='zstd',
        compression_level=5,
        use_pyarrow=True,
    )
    
def align_order(df_1: pl.DataFrame, df_2: pl.DataFrame, on: str) -> pl.DataFrame:
    """
    Aligns df_2 to the order of df_1 based on a common column.

    Parameters:
    - df_1: The reference DataFrame whose order will be followed.
    - df_2: The DataFrame to be reordered.
    - on: The column name used for alignment.

    Returns:
    - A new DataFrame with rows of df_2 ordered to match df_1.
    """
    # Add row number to preserve df_1's order
    df_1_with_order = df_1.select([
        pl.col(on),
        pl.int_range(pl.len()).alias("__order__")
    ])
    
    # Join df_2 with the order information and sort
    return (
        df_2
        .join(df_1_with_order, on=on, how="inner")
        .sort("__order__")
        .drop("__order__")
    )
    