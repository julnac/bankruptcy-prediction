from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import pandas as pd
import numpy as np

calculated_features = [
    # Profitability Metrics
    'Net Profit Margin percentage', 'EBITDA Margin percentage', 'Gross Profit Margin percentage', 'ROA percentage',
    # Liquidity & Solvency Ratios
    'Current Ratio', 'Quick Ratio', 'Liabilities-to-Assets Ratio', 'Total Equity', 'Equity-to-Asset',
    # Efficiency Ratios
    'Asset Turnover', 'Receivables Turnover', 'Inventory Turnover',
    # Valuation Metrics
    'PE Ratio'
]


def calculate_ratios(df, fill_na=True):
    # Profitability Metrics
    df['Net Profit Margin percentage'] = np.where(df['Revenue'] != 0, df['Net Income'] / df['Revenue'], np.nan) * 100
    df['EBITDA Margin percentage'] = np.where(df['Revenue'] != 0, df['EBITDA'] / df['Revenue'], np.nan) * 100
    df['Gross Profit Margin percentage'] = np.where(df['Revenue'] != 0, df['Gross Profit'] / df['Revenue'],
                                                    np.nan) * 100
    df['ROA percentage'] = np.where(df['Total Assets'] != 0, df['Net Income'] / df['Total Assets'], np.nan) * 100

    # Liquidity & Solvency Ratios
    df['Current Ratio'] = np.where(df['Total Current Liabilities'] != 0,
                                   df['Total Current Assets'] / df['Total Current Liabilities'], np.nan)
    df['Quick Ratio'] = np.where(df['Total Current Liabilities'] != 0,
                                 (df['Total Current Assets'] - df['Total Inventories']) / df[
                                     'Total Current Liabilities'], np.nan)
    df['Liabilities-to-Assets Ratio'] = np.where(df['Total Assets'] != 0, df['Total Liabilities'] / df['Total Assets'],
                                                 np.nan)
    df['Total Equity'] = df['Total Assets'] - df['Total Liabilities']
    df['Equity-to-Asset'] = np.where(df['Total Assets'] != 0, df['Total Equity'] / df['Total Assets'], np.nan)

    # Efficiency Ratios
    df['Asset Turnover'] = np.where(df['Total Assets'] != 0, df['Revenue'] / df['Total Assets'], np.nan)
    df['Receivables Turnover'] = np.where(df['Total Receivables'] != 0, df['Revenue'] / df['Total Receivables'],
                                          np.nan)  # Approximation, as average is not available
    df['Inventory Turnover'] = np.where(df['Total Inventories'] != 0,
                                        df['Cost of Goods Sold'] / df['Total Inventories'],
                                        np.nan)  # Approximation, as average is not available

    # Valuation Metrics
    df['PE Ratio'] = np.where(df['Net Income'] != 0, df['Market Cap'] / df['Net Income'], np.nan)

    if fill_na:
        df[calculated_features] = df[calculated_features].fillna(0)

    df[calculated_features] = df[calculated_features].round(2)

    return df


def get_classifiers(random_state=43, verbose=0):
    lgbmClassifierVerbose = -1 if verbose == 0 else 1
    
    # https://scikit-learn.org/stable/supervised_learning.html
    # Linear Models
    logisticRegression = LogisticRegression(random_state=random_state, max_iter=300, verbose=verbose)
    # Support Vector Machines
    svc = SVC(random_state=random_state, verbose=verbose)
    # Stochastic Gradient Descent
    sgdClassifier = SGDClassifier(random_state=random_state, verbose=verbose)
    # Nearest Neighbors
    kNeighborsClassifier = KNeighborsClassifier()
    # Naive Bayes
    gaussianNB = GaussianNB()
    # Decision Trees
    decisionTreeClassifier = DecisionTreeClassifier(random_state=random_state)
    # Ensembles: Gradient boosting, random forests, bagging, voting, stacking
    randomForestClassifier = RandomForestClassifier(random_state=random_state, verbose=verbose)
    baggingClassifier = BaggingClassifier(random_state=random_state, verbose=verbose)
    adaBoostClassifier = AdaBoostClassifier(algorithm='SAMME', random_state=random_state)
    extraTreesClassifier = ExtraTreesClassifier(random_state=random_state, verbose=verbose)

    # Other popular classifiers
    lgbmClassifier = LGBMClassifier(random_state=random_state, verbose=lgbmClassifierVerbose)
    xgbClassifier = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss', verbosity=verbose)

    return [
        logisticRegression,
        svc,
        sgdClassifier,
        kNeighborsClassifier,
        gaussianNB,
        decisionTreeClassifier,
        randomForestClassifier,
        baggingClassifier,
        adaBoostClassifier,
        extraTreesClassifier,
        lgbmClassifier,
        xgbClassifier]


def flatten_financial_dataset(financial_dataset, object_length_in_rows=5):
    metadata_columns_length = 2

    metadata_columns = list(financial_dataset.columns[:metadata_columns_length].values)
    # 'label', 'subset'

    value_columns = financial_dataset.columns[metadata_columns_length:]
    new_columns = metadata_columns + [f'{col}_{i + 1}' for i in range(object_length_in_rows) for col in value_columns]
    # ['label', 'subset'] + [f'{col}_{i + 1}' for i in range(object_length_in_rows) for col in value_columns]
    dfs = []

    for i in range(0, len(financial_dataset), object_length_in_rows):
        # Group the dataset into chunks of `object_length_in_rows`
        group = financial_dataset.iloc[i:i + object_length_in_rows]
        if len(group) < object_length_in_rows:
            raise ValueError(
                f"Dataset is not properly structured for flattening. Expected {object_length_in_rows} rows per object, "
                f"but found {len(group)} rows."
            )

        # df.iloc[row_indexer, column_indexer]
        # Extract metadata columns from the first row of the group
        label = group['label'].iloc[0]
        subset = group['subset'].iloc[0]

        # Flatten the values of the group, excluding metadata columns
        # group.drop(columns=metadata_columns) removes the metadata columns from the group
        # .values.flatten() converts the DataFrame to a 1D numpy array - lata po sobie w jednym wierszu
        values = group.drop(columns=metadata_columns).values.flatten()

        dfs.append([label, subset] + values.tolist())

    final_flatten_df = pd.DataFrame(dfs, columns=new_columns)
    final_flatten_df = final_flatten_df.reset_index(drop=True)
    return final_flatten_df


def deflatten_financial_dataset(flatten_df, object_length_in_rows=5):
    metadata_columns_length = 2

    values_columns = flatten_df.columns[metadata_columns_length:]
    unique_values_columns = sorted(set(col.split('_')[0] for col in values_columns))

    new_columns = flatten_df.columns[:metadata_columns_length].tolist() + unique_values_columns

    deflattened_rows = []

    for _, row in flatten_df.iterrows():
        first_columns = row.values[:metadata_columns_length]
        # fiscal_periods = row.values[4].split(';')

        reshaped_values = np.array(row.values[metadata_columns_length:]).reshape(object_length_in_rows, -1)

        for i in range(object_length_in_rows):
            # new_row = np.concatenate([first_columns[:1], [fiscal_periods[i]], reshaped_values[i]])
            new_row = np.concatenate([first_columns[:metadata_columns_length], reshaped_values[i]])
            deflattened_rows.append(new_row)

    deflatten_df = pd.DataFrame(deflattened_rows, columns=new_columns)

    return deflatten_df.reset_index(drop=True)


def split_train_test(df, random_state, test_size, object_length_in_rows=5):
    df = df.copy()
    # dodaje kolumne z indeksami przypisanymi do grupy (pierwsza grupa ma indeksy 0 itd.)
    df['object_id'] = df.index // object_length_in_rows

    # pierwsze obiekty z grupy
    objects = df.groupby('object_id').first().reset_index()
    # wartości labels dla każdego pierwszego obiektu(spółki) w grupie
    labels = objects['label'].values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, test_idx = next(splitter.split(objects, labels))

    # test_object_ids = objects.loc[test_idx, 'object_id']

    #worzymy nową kolunę 'subset' i przypisujemy wartości 'train' lub 'test' w zależności od tego, czy object_id jest w test_idx
    df['subset'] = 'train'
    df.loc[df['object_id'].isin(test_idx), 'subset'] = 'test'

    df = df.drop('object_id', axis=1)

    # position subset column after 'label'
    subset_col = df.pop('subset')
    df.insert(df.columns.get_loc('label') + 1, 'subset', subset_col)

    return df


def oversample(X, y, random_state):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def oversample_train_subset(dataset, random_state):
    # nadpróbkowujemy tylko dane treningowe
    train_df = dataset[dataset['subset'] == 'train'].drop('subset', axis=1)
    X = train_df.drop('label', axis=1)
    y = train_df['label']

    X_train, y_train = oversample(X, y, random_state)
    X_train = X_train.round(1)

    # tworzymy nowy dataframe z nadpróbkowanymi danymi
    train_oversampled = pd.DataFrame(X_train, columns=X.columns)
    train_oversampled['label'] = y_train
    train_oversampled['subset'] = 'train'

    # dodajemy oversampled train data do test data
    test_df = dataset[dataset['subset'] == 'test']
    test_df = test_df.reset_index(drop=True)
    result_df = pd.concat([train_oversampled, test_df], ignore_index=True)

    # ustawiamy kolumny, aby 'label' i 'subset' były na początku
    cols = ['label', 'subset'] + [col for col in result_df.columns if col not in ['label', 'subset']]
    result_df = result_df[cols]
    result_df = result_df.reset_index(drop=True)

    return result_df


def get_train_test_split(X, y):
    X_train = X[X['subset'] == 'train']
    y_train = y[X['subset'] == 'train']

    X_test = X[X['subset'] == 'test']
    y_test = y[X['subset'] == 'test']

    X_train = X_train.drop('subset', axis=1)
    X_test = X_test.drop('subset', axis=1)

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    return X_train, y_train, X_test, y_test


def get_data(dataset, flatten, enriched, oversampled, shuffled=True, scaled=True, random_state=2025, object_length_in_rows=5, test_size=0.1):
    # dodajemy nową kolumnę 'subset' z wartością 'train' lub 'test'
    dataset = split_train_test(dataset, random_state, test_size)

    # dodajemy więcej danych do training setu, tak aby zbiór był zbalansowany
    if oversampled:
        flatten_dataset = flatten_financial_dataset(dataset, object_length_in_rows)
        flatten_dataset = oversample_train_subset(flatten_dataset, random_state)
        dataset = deflatten_financial_dataset(flatten_dataset)

    # obliczamy i dodajemy wskaźniki finansowe
    if enriched:
        dataset = calculate_ratios(dataset)

    # spłaszczamy dane, aby każda spółka była w jednym wierszu
    if flatten:
        dataset = flatten_financial_dataset(dataset, object_length_in_rows)

    # losowa kolejność wierszy
    if shuffled:
        if not flatten:
            dataset = flatten_financial_dataset(dataset, object_length_in_rows)
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        if not flatten:
            dataset = deflatten_financial_dataset(dataset)
        
    X_train, y_train, X_test, y_test  = get_train_test_split(dataset.drop('label', axis=1), dataset['label'])

    if scaled:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
