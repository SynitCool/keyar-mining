import pandas as pd

URL = "https://raw.githubusercontent.com/SynitCool/Data-Science/main/Covid19%20Indonesia/Clustering/Dataset/Cleaned/Covid19propinsi_Cleaned.csv"

def make_covid_id():
    """
    make dataset covid 19 indonesia in link
    https://raw.githubusercontent.com/SynitCool/Data-Science/main/Covid19%20Indonesia/Clustering/Dataset/Cleaned/Covid19propinsi_Cleaned.csv

    Returns
    -------
    df : pandas.DataFrame
        returning as pandas DataFrame.

    """
    df = pd.read_csv(URL)
    
    return df

def get_example_covid_id():
    """
    make example dataset

    Returns
    -------
    df : pandas.DataFrame
        returning as pandas DataFrame that has been selected.

    """
    
    df = make_covid_id()
    
    columns = ["Daily_Case", "Daily_Death"]
    
    df = df[columns]
    
    return df