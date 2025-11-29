import os
import pandas as pd
import pandas_gbq


class DataLoader():
    ''' The main source of training data.
    
    The class provides a standarised method to query actual training data
    from BigQuery, and to easily log the used dataset, and parameters to Mlflow.
    '''
    def __init__(self, start_date: str = None, end_date: str = None, mock: bool = False):
        ''' Init the training data loader with query settings.
        
        Parameters
        ----------
        start_date: str, optional
            The minimum date to query data. If none, no filtter is applied
        end_date: str, optional
            The last date to query data. If none, no filtter is applied
        mock: bool, default=False
            If true, a small sample DataFrame is returned instead of querying
            the actual source data.
        '''
        self.__start_date = start_date
        self.__end_date = end_date
        self.__mock = mock


    @property
    def sql(self) -> str:
        """ Return the used query to pull training data as string
        
        The query is always the same since the source data does not change,
        but the user can specify the time range, and sorting the of the data.

        Returns
        -------
        sql: str
            The query used to pull data with dynamic options
        """
        start_filtter = f"AND KeyDate >= '{self.__start_date}'" if self.__start_date else ''
        end_filtter = f"AND KeyDate <='{self.__end_date}'" if self.__end_date else ''

        sql = f"""
        SELECT
            KeyDate AS date,
            Receiver AS receiver,
            Amount AS amount,
            Category AS category
        FROM
            {os.getenv('GCP_BQ_DATASET')}.f_transactions
        WHERE
            category != 'N/A'
            AND receiver is not NULL
            {start_filtter}
            {end_filtter}
        ORDER BY
            date asc
        """
        return sql


    def load(self) -> pd.DataFrame:
        """ The main method to load the trainig data.
        
        Querys the source table with the user filtter settings.
        If mock mode is enabled, a small sample DataFrame is returned instead.

        Returns
        -------
        df: pd.DataFrame
            The queryed training data, that has bee sorted by date, and Nans dropped.
        """
        if self.__mock:
            return self.__mock_load()

        df = pandas_gbq.read_gbq(self.sql, 
                                project_id=os.getenv('GCP_PROJECT_ID'),
                                location=os.getenv('GCP_LOCATION'), 
                                progress_bar_type=None) # Use default Account/Cloud Run SA
        
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        df = df.dropna().reset_index(drop=True)
        return df


    def __mock_load(self) -> pd.DataFrame:
        """ A mock method to load sample data for testing purposes.
        
        Returns
        -------
        df: pd.DataFrame
            A small sample DataFrame for testing purposes.
        """
        data = {
            'date': [
            '2023-01-01', '2023-01-02', '2023-01-03',
            '2023-01-05', '2023-01-07', '2023-01-10',
            '2023-01-12', '2023-01-15', '2023-01-17',
            '2023-01-20', '2023-01-22', '2023-01-25',
            '2023-01-28', '2023-02-01', '2023-02-03'
            ],
            'receiver': [
            'K Market Helsinki Railway centre',
            'Big Corporation Helsinki',
            'Big Cinema City Helsinki',
            'Uber Eats Finland',
            'Siemens Energy Oy',
            'Electricity Co Helsinki',
            'Helsinki Public Transport',
            'Nordic Gym Helsinki',
            'Pharmacy Helsinki Central',
            'Apartment Rent Oy',
            'Salary Payout Ltd',
            'Online Bookstore',
            'Refund - Electronics Store',
            'Coffee Corner Helsinki',
            'Supermarket East Side'
            ],
            'amount': [
            -43.34, 3201.87, -15.00,
            -28.50, -1200.00, -78.90,
            -2.80, -49.00, -12.99,
            -950.00, 3201.87, -24.95,
            150.00, -3.40, -62.10
            ],
            'category': [
            'FOOD', 'SALARY', 'ENTERTAINMENT',
            'FOOD', 'SERVICES', 'BILLS',
            'TRANSPORT', 'HEALTH', 'HEALTH',
            'RENT', 'SALARY', 'SHOPPING',
            'REFUND', 'FOOD', 'GROCERIES'
            ],
        }
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').dt.date
        return df