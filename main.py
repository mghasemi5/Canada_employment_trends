import functions

df_init = functions.read_data('employment_trends.csv')
df_init = functions.remove_empty_columns(df_init)
df_job, df_sal = functions.split_dataframe_by_uom(df_init)
df_sal,df_job = functions.impute_missing_values(df_sal,df_job)
industries= ['Trade','Management of companies and enterprises','Real estate and rental and leasing','Manufacturing','Mining, quarrying, and oil and gas extraction','Goods producing industries','Professional, scientific and technical services']
functions.plot_salary_trends_by_industry(df_sal,industries)