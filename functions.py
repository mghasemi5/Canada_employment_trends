import pandas as pd


def read_data(address):
    return pd.read_csv(address)


def describe_data(data):
    return data.describe()


def remove_empty_columns(df):
    """
    Remove columns that are entirely empty in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with empty columns removed.
    """
    # Identify columns with all NaN values
    empty_columns = df.columns[df.isnull().all()]

    # Drop the empty columns
    df_cleaned = df.drop(columns=empty_columns)

    print(f"Removed columns: {list(empty_columns)}")
    return df_cleaned


def split_dataframe_by_uom(df):
    """
    Split a DataFrame into two based on the UOM column.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: Two DataFrames, one for employment rate (UOM='Persons') and one for salary rate (UOM='Dollars').
    """
    # Filter rows based on the 'UOM' column
    employment_df = df[df['UOM'] == 'Persons'].reset_index(drop=True)
    salary_df = df[df['UOM'] == 'Dollars'].reset_index(drop=True)

    print(f"Employment rate DataFrame shape: {employment_df.shape}")
    print(f"Salary rate DataFrame shape: {salary_df.shape}")

    return employment_df, salary_df


def impute_missing_values(salary_df, employment_df):
    """
    Impute missing values in the VALUE column for salary and employment DataFrames.

    Parameters:
        salary_df (pd.DataFrame): DataFrame for salary rate (UOM='Dollars').
        employment_df (pd.DataFrame): DataFrame for employment rate (UOM='Persons').

    Returns:
        tuple: Two DataFrames with missing values imputed.
    """
    # Impute missing values in salary DataFrame with column mean
    salary_df['VALUE'] = salary_df['VALUE'].fillna(salary_df['VALUE'].mean())

    # Impute missing values in employment DataFrame with 0
    employment_df['VALUE'] = employment_df['VALUE'].fillna(0)

    print("Missing values imputed:")
    print(f"Salary DataFrame - VALUE missing count: {salary_df['VALUE'].isnull().sum()}")
    print(f"Employment DataFrame - VALUE missing count: {employment_df['VALUE'].isnull().sum()}")

    return salary_df, employment_df


def plot_employment_trend(employment_df):
    """
    Plot the total employment trend over time with x-axis ranging from 2001 to 2024.

    Parameters:
        employment_df (pd.DataFrame): DataFrame containing employment data.
    """
    import matplotlib.pyplot as plt

    # Group data by year and sum the employment values
    yearly_data = employment_df.groupby(employment_df['REF_DATE'].str[:4])['VALUE'].sum()

    # Convert index to integer for better x-axis control
    yearly_data.index = yearly_data.index.astype(int)

    plt.figure(figsize=(12, 6))
    plt.plot(yearly_data.index, yearly_data.values, marker='o', label='Total Employment')

    # Set x-axis range
    plt.xlim(2001, 2023)
    plt.xticks(range(2001, 2024, 1), rotation=45)

    # Labels and title
    plt.title('Total Employment Trend Over Time (2001–2024)')
    plt.xlabel('Year')
    plt.ylabel('Total Employment')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_employment_distribution_by_month(employment_df):
    """
    Plot the distribution of employment data by month as a histogram,
    where the height of each bar represents the sum of VALUE for that month.

    Parameters:
        employment_df (pd.DataFrame): DataFrame containing employment data.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Extract month from REF_DATE and calculate the sum of VALUE for each month
    employment_df['Month'] = pd.to_datetime(employment_df['REF_DATE']).dt.month
    monthly_totals = employment_df.groupby('Month')['VALUE'].sum()

    plt.figure(figsize=(10, 6))
    plt.plot(monthly_totals.index, monthly_totals.values, color='red')

    # Labels and title
    plt.title('Employment Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Total Employment')
    plt.xticks(range(1, 13),
               ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_employment_by_industry(employment_df):
    """
    Plot the distribution of employment data by industry, with improved readability.

    - Shortens industry names by splitting at '[' and removing job codes.
    - Excludes the first two rows from the chart.

    Parameters:
        employment_df (pd.DataFrame): DataFrame containing employment data.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Exclude the first two rows
    filtered_df = employment_df.iloc[2:]
    # Group data by industry and sum employment values
    industry_data = filtered_df.groupby('North American Industry Classification System (NAICS)')['VALUE'].sum().reset_index()

    # Process industry names to exclude job codes
    industry_data['NAICS_Short'] = industry_data['North American Industry Classification System (NAICS)'].str.split('[').str[0]

    # Sort by total employment for better visualization
    industry_data = industry_data.sort_values(by='VALUE', ascending=False)[2:]

    # Create the bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(y='NAICS_Short', x='VALUE', data=industry_data, palette='viridis')

    # Adjust font sizes for better readability
    plt.title('Employment Distribution by Industry', fontsize=14)
    plt.xlabel('Total Employment', fontsize=12)
    plt.ylabel('Industry', fontsize=12)
    plt.yticks(fontsize=5,rotation=30)  # Smaller font size for y-axis labels
    plt.show()

def plot_employment_by_region(employment_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    region_data = employment_df.groupby('GEO')['VALUE'].sum().reset_index()
    region_data = region_data.sort_values(by='VALUE', ascending=False)
    sns.barplot(x='VALUE', y='GEO', data=region_data)
    plt.title('Employment Distribution by Region')
    plt.xlabel('Total Employment')
    plt.ylabel('Region')
    plt.yticks(rotation=45)
    plt.show()

def compare_employment_across_industries(employment_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.boxplot(y='North American Industry Classification System (NAICS)', x='VALUE', data=employment_df)
    plt.title('Comparison of Employment Levels Across Industries')
    plt.xlabel('Employment')
    plt.ylabel('Industry')
    plt.show()


def plot_salary_trends(salary_df):
    """
    Plot the average salary trend over time.

    Parameters:
        salary_df (pd.DataFrame): DataFrame containing salary data.
    """
    import matplotlib.pyplot as plt

    # Ensure REF_DATE is a string (convert if necessary)
    salary_df['REF_DATE'] = salary_df['REF_DATE'].astype(str)

    # Group by year and calculate the mean salary
    yearly_data = salary_df.groupby(salary_df['REF_DATE'].str[:4])['VALUE'].mean()

    # Convert index to integer for proper plotting
    yearly_data.index = yearly_data.index.astype(int)

    # Plot the trend
    plt.figure(figsize=(10, 6))
    plt.plot(yearly_data.index, yearly_data.values, marker='o', label='Average Salary')

    # Set x-axis range and labels
    plt.xlim(2001, 2024)
    plt.xticks(range(2001, 2024, 1), rotation=45)
    plt.xlabel('Year')
    plt.ylabel('Average Salary')
    plt.title('Average Salary Trend Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_salary_across_states(salary_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.boxplot(x='GEO', y='VALUE', data=salary_df)
    plt.title('Salary Comparison for Similar Jobs Across States')
    plt.xlabel('State')
    plt.ylabel('Salary')
    plt.xticks(rotation=20,fontsize=8)
    plt.show()

def compare_salary_by_industry(salary_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    industry_data = salary_df.groupby('North American Industry Classification System (NAICS)')['VALUE'].mean().reset_index()
    industry_data['NAICS_Short'] = \
    industry_data['North American Industry Classification System (NAICS)'].str.split('[').str[0]
    industry_data = industry_data.sort_values(by='VALUE', ascending=False)
    sns.barplot(x='VALUE', y='NAICS_Short', data=industry_data)
    plt.title('Salary Comparison by Industry')
    plt.yticks(fontsize=5,rotation=30)
    plt.xlabel('Average Salary')
    plt.ylabel('Industry')
    plt.show()

def plot_correlation_between_employment_and_salary(employment_df, salary_df):
    import seaborn as sns
    import matplotlib.pyplot as plt
    combined_df = pd.merge(employment_df, salary_df, on=['REF_DATE', 'GEO', 'North American Industry Classification System (NAICS)'], suffixes=('_employment', '_salary'))
    sns.scatterplot(x='VALUE_employment', y='VALUE_salary', data=combined_df)
    plt.title('Correlation Between Employment Levels and Salaries')
    plt.xlabel('Employment Levels')
    plt.ylabel('Salaries')
    plt.show()


def plot_multiple_industry_trends(employment_df, industry_names):
    """
    Plot employment trends for multiple industries on the same chart.

    Parameters:
        employment_df (pd.DataFrame): DataFrame containing employment data.
        industry_names (list): List of up to 5 industry names to plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    for industry_name in industry_names:
        # Filter data for the specific industry
        filtered_data = employment_df[
            employment_df['North American Industry Classification System (NAICS)'].str.contains(industry_name,
                                                                                                na=False)]

        # Group by year and calculate total employment
        yearly_data = filtered_data.groupby(filtered_data['REF_DATE'].str[:4])['VALUE'].sum()
        yearly_data.index = yearly_data.index.astype(int)

        # Plot the trend for this industry
        plt.plot(yearly_data.index, yearly_data.values, marker='o', label=industry_name)

    # Chart customization
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Total Employment', fontsize=12)
    plt.title('Employment Trends for Selected Industries', fontsize=14)
    plt.legend(title='Industries', fontsize=10,loc='best')
    plt.grid(True)
    plt.xticks(range(yearly_data.index.min(), yearly_data.index.max() + 1, 2), rotation=45)
    plt.show()


def plot_salary_trends_by_industry(salary_df, industry_names):
    """
    Plot salary trends over time for selected industries.

    Parameters:
        salary_df (pd.DataFrame): DataFrame containing salary data.
        industry_names (list): List of industries to include in the plot.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))

    for industry_name in industry_names:
        # Filter data for the specific industry
        filtered_data = salary_df[
            salary_df['North American Industry Classification System (NAICS)'].str.contains(industry_name, na=False)]

        # Group by year and calculate average salary
        yearly_data = filtered_data.groupby(filtered_data['REF_DATE'].str[:4])['VALUE'].mean()
        yearly_data.index = yearly_data.index.astype(int)

        # Plot the trend for this industry
        plt.plot(yearly_data.index, yearly_data.values, marker='o', label=industry_name)

    # Chart customization
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Salary', fontsize=12)
    plt.title('Salary Trends by Industry', fontsize=14)

    # Align the legend
    plt.legend(title='Industries', fontsize=10, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

    plt.grid(True)
    plt.xticks(range(yearly_data.index.min(), yearly_data.index.max() + 1, 2), rotation=45)
    plt.tight_layout()
    plt.show()
