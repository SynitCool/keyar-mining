import pandas as pd


def make_fruit_sold_dummy_association_rules():
    """
    functions that make dummy dataframe

    Returns
    -------
    df : pandas.DataFrame
        returning as df.

    """

    # Make dataset
    df = pd.DataFrame(
        {
            "InvoiceNo": [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "Description": [
                "apple",
                "banana",
                "mango",
                "banana",
                "mango",
                "apple",
                "strawberry",
                "strawberry",
                "mango",
                "banana",
            ],
            "Country": ["USA" for _ in range(10)],
            "Quantity": [3, 2, 6, 8, 5, 4, 3, 2, 7, 4],
        }
    )

    # Cleaning dataset
    clean_df = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .reset_index()
        .fillna(0)
        .set_index("InvoiceNo")
    )

    clean_df = clean_df.applymap(lambda x: 1 if x > 0 else 0)

    return clean_df


def make_dummy_data_classification():
    """
    making dummy data for classification

    Returns
    -------
    df : pandas.DataFrame
        returning as df.
    """

    clf_data = [
        [22, 1],
        [23, 1],
        [21, 1],
        [18, 1],
        [19, 1],
        [25, 0],
        [27, 0],
        [29, 0],
        [31, 0],
        [45, 0],
    ]

    return pd.DataFrame(clf_data)
