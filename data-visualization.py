# Visualize the generated revenue by month and year

sales_m_y = pd.DataFrame(sales_train['revenue'].groupby([sales_train['date_block_num']]).agg('sum'))
sales_m_y = sales_m_y.reset_index()
#sales_m_y.head()
sales_my_plot = sns.relplot(x = 'date_block_num', y = 'revenue', data = sales_m_y, kind='line')
sales_my_plot.set(xlabel = 'Month & Year', ylabel ='Total Revenue')
plt.show()
