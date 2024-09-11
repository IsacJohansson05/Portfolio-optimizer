import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

plt.style.use('seaborn-v0_8-dark')

class ModernPortfolioTheory:
    """Uses modern portfolio theory to uptimize weight for each asset in a portfolio"""
    def __init__(self,portfolio,end_year=2024,years=10):
        """Initializes"""
        self.portfolio=portfolio
        self.end_year=end_year
        self.years=years


    def get_close_data(self):
        """Extracts close data from avanza API and returns it in a pandas time series data frame"""
        portfolio=self.portfolio
        end_year=self.end_year
        years=self.years

        # Creates an empty data frame where we will store close data for all assets
        close=pd.DataFrame()

        for asset, id in portfolio.items():

            # Start with an empty array for all assets
            his_data_array=[]


            # Downloads data for every year in selected time period for one asset while storing historical data in an array, finally puts them together in a data frame
            for year in range(end_year-years,end_year):
                response=requests.get(f'https://www.avanza.se/_api/fund-guide/chart/{id}/{year}-01-01/{year}-12-31?raw=true')

                
                
                # The response contains a data series from the avanza API where time (x) is described in milliseconds, we convert this into CEST time
                try:
                    one_year_data=pd.DataFrame(response.json()['dataSerie'])
                    one_year_data.set_index(pd.to_datetime(one_year_data.pop('x'),unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin'),inplace=True)
                
                
                # Handles exceptions since the security might not have data for a selected year
                except:
                    print(f"Failed to download data for {asset} in year {year}")

                # All data for ONE asset gets dumped inside an array
                his_data_array.append(one_year_data.y)
            
            # When the array is filled with data from all years, we put them together in the y axis it inside the dataframe previously created
            close[asset]=pd.concat(his_data_array)

        # Turns the index row to date format (previously yy-mm-dd hh:mm:ss) and drops rows where atleast on asset lacks closing data
        close.index=close.index.date
        close.index.name="date"
        close.dropna(inplace=True)
        return close
    
    def get_index_close_data(self):
        """Gets data for popular indexes through avanza api"""
        end_year=self.end_year
        years=self.years

        index_df=pd.DataFrame()
        spy_array=[]
        omx_array=[]
        global_array=[]

        for year in range(end_year-years,end_year):

            spy_response=requests.get(f'https://www.avanza.se/_api/fund-guide/chart/159932/{year}-01-01/{year}-12-31?raw=true')
            spy_one_year_data=pd.DataFrame(spy_response.json()['dataSerie'])
            spy_one_year_data.set_index(pd.to_datetime(spy_one_year_data.pop('x'),unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin'),inplace=True)
            spy_array.append(spy_one_year_data.y)

            omx_response=requests.get(f'https://www.avanza.se/_api/fund-guide/chart/19002/{year}-01-01/{year}-12-31?raw=true')
            omx_one_year_data=pd.DataFrame(omx_response.json()['dataSerie'])
            omx_one_year_data.set_index(pd.to_datetime(omx_one_year_data.pop('x'),unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin'),inplace=True)

            omx_array.append(omx_one_year_data.y)

            global_response=requests.get(f'https://www.avanza.se/_api/fund-guide/chart/155324/{year}-01-01/{year}-12-31?raw=true')
            global_one_year_data=pd.DataFrame(global_response.json()['dataSerie'])
            global_one_year_data.set_index(pd.to_datetime(global_one_year_data.pop('x'),unit='ms').dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin'),inplace=True)

            global_array.append(global_one_year_data.y)
        
        index_df['SPY']=pd.concat(spy_array)
        index_df['OMX']=pd.concat(omx_array)
        index_df['GLOBAL']=pd.concat(global_array)

        index_df.index=index_df.index.date
        index_df.index.name="date"
        index_df.dropna(inplace=True)
        return index_df
    
    def get_change_data(self):
        """Returns a data frame for price change"""
        close=self.get_close_data()
        change=(close/close.shift(1)-1).dropna() # Could also be done by close.pct_change().dropna()

        return change
        
    def get_mean_std(self):
        """Returns a transposed data frame that indicates historical annual volatility and returns"""
        change=self.get_change_data()

        # Gets the volatility and returns from the daily change by using .describe(), transpose it to only extract the columns std and mean
        mean_std=change.describe().T.loc[:,["mean","std"]]
        # Annualize (on average 252 trading days per year)
        mean_std["mean"]=mean_std["mean"]*252
        mean_std["std"]=mean_std["std"]*np.sqrt(252)

        return mean_std
        

    
    def get_covariance(self):
        """Returns annualized covariance by using the .cov() funcion and multiplying my number of trading days"""
        cov=252*self.get_change_data().cov()
        
        return cov

    def get_efficient_frontier(self,rf=0.019):
        """Returns an data frame of the efficient frontier"""
        mean=np.array(self.get_mean_std()['mean']) # Mean return vector
        cov_matrix=np.array(self.get_covariance()) # Covariance square matrix

        
        portfolio_std_mean_list=[] # A list that stores array of porfolios std and mean
        weight_array_list=[] # A list that stores weight arrays

        allocation_df = pd.DataFrame(columns=self.get_close_data().columns)

        
        for i in range(100000): # More than 1M allocations usually takes long
            """A loop that tests different allocations and calculates its annual volatility and return"""
            
            # Makes random numbers that adds up to 1
            random_array=np.random.rand(len(mean))
            weight_array=random_array/random_array.sum()

            # Calculates the expected outcome (correct matrix multiplication operation by using np.dot NOT *)
            portfolio_std=np.sqrt(np.dot(weight_array,np.dot(weight_array.T,cov_matrix)))
            portfolio_mean=np.dot(mean,weight_array)
            
            portfolio_std_mean=[portfolio_std,portfolio_mean]
            
            # Stores the arrays inside the lists
            portfolio_std_mean_list.append(portfolio_std_mean)
            weight_array_list.append(weight_array)

        # Makes pandas data frames of the results
        allocation_df = pd.DataFrame(weight_array_list,columns=self.portfolio.keys())
        mean_std_df = pd.DataFrame(portfolio_std_mean_list, columns=["std", "ret"])

        # Puts them together in the x-asis
        results_df=pd.concat([allocation_df,mean_std_df],axis=1)

        # Calculates sharpe ratio of each allocation by using a pre-determined risk free interest rate (rf)
        results_df['Sharpe Ratio']=(results_df['ret']-rf)/results_df['std']
        


        
        return results_df
    
 

    def get_best_allocation(self, rf=0.019):
        """Scatter plots the efficient frontier and returns the optimal allocation"""
        
        results_df = self.get_efficient_frontier(rf)

        # Optimal allocation where the sharpe ratio is max
        optimal_allocation = results_df.loc[results_df['Sharpe Ratio'] == results_df['Sharpe Ratio'].max()]

        # Calculates returns for the all assets concidering it's weight
        for symbol in self.portfolio.keys():
            weighted_ret=optimal_allocation[symbol].values[0]*self.get_change_data()
        
        # Calculates portfolio return by summing them
        portfolio_ret=weighted_ret.sum(axis=1)
        
        # Calculates sortinos ratio
        losses=portfolio_ret[portfolio_ret<0]
        
        std_loss=losses.std()
        std_loss*=np.sqrt(252)

        optimal_allocation['Sortino Ratio']=(optimal_allocation['ret']-rf)/std_loss



        



        
        fig, ax = plt.subplots(figsize=(11, 7))

        # Creates the scatter of all allocations
        scatter = ax.scatter(
            results_df["std"], 
            results_df["ret"], 
            c=results_df["Sharpe Ratio"], 
            cmap="winter", 
            marker="o", 
            label="Portfolio Allocations"
        )

        # Makes a colorbar for the scatter where greener indicates higher sharpe ratio
        cbar = plt.colorbar(scatter)
        cbar.set_label("Sharpe Ratio")
        
        # Makes a red dot for optimal allocation
        ax.scatter(
            optimal_allocation["std"], 
            optimal_allocation["ret"], 
            color="red", 
            marker="o", 
            label="Optimal Allocation"
        )

        # Customizes the plot apperance
        ax.set_title("Risk to Volatility for Different Portfolio Allocations")
        ax.set_xlabel("Annual Volatility (std)")
        ax.set_ylabel("Expected Annual Return (mean)")
        ax.grid(True)
        ax.legend()

        

        return optimal_allocation,fig

    def get_buy_hold_performance(self,rf=0.019):
        """Calculates the cumulative performance of the optimal portfolio using buy and hold strategy."""
        close = self.get_close_data()
        log_ret = np.log(close / close.shift(1))  # Log returns for the assets

        # Collects optimal allocations
        optimal_allocation = self.get_best_allocation(rf)[0]

        # Initializes data frame that holds weighted log ret
        log_ret_weighted = pd.DataFrame()

        # Apply the optimal weights to each asset's log return
        for column in log_ret.columns:
            log_ret_weighted[column] = optimal_allocation[column].values[0] * log_ret[column]

        # Sum the weighted log returns to get the portfolio's total log return
        portfolio_df = pd.DataFrame()
        portfolio_df["log_ret"] = log_ret_weighted.sum(axis=1) # x-axis

        # Calculate the cumulative log return (i.e., the buy-and-hold strategy)
        portfolio_df["cum_log_ret"] = portfolio_df["log_ret"].cumsum()

        # Convert cumulative log return to normal returns
        portfolio_buy_hold = np.exp(portfolio_df["cum_log_ret"])

        return portfolio_buy_hold


    def compare_to_index(self,rf):
        """Compares portfolio performance to the market"""
        df = self.get_index_close_data()
        
        df = df.div(df.iloc[0])  # Normalize the data to start at the same level (alternative to log route)

        mean_ret_indexes=pd.DataFrame(df.copy().pct_change().mean())
        ret_indexes=df.copy().pct_change().dropna()


        df["Optimized Portfolio"] = self.get_buy_hold_performance(rf)
        ret_optimized=df["Optimized Portfolio"].pct_change().dropna()

        mean_ret_optimized=df["Optimized Portfolio"].pct_change().mean()

        returning_df=pd.DataFrame()
        returning_df["Annual Outperformance (Alpha % units)"]=(mean_ret_optimized-mean_ret_indexes)*100*252

        # Calculates beta to all indexes note. covariance is a matrix
        beta_dict={}
        for column in ret_indexes.columns:
            cov_matrix=np.cov(ret_optimized,ret_indexes[column])
            cov=cov_matrix[0,1]
            var=cov_matrix[1,1]
            beta=cov/var
            beta_dict[column]=beta

        returning_df["Market Sensitivity (Beta)"]= beta_dict            

        




        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(11, 7))

        
        df.plot(ax=ax,title="$1 buy and hold compared to popular indexes",xlabel="Date",ylabel="Normalized Value",grid=True)  

        
        
        return fig,returning_df  # Return the figure to be rendered in Streamlit


        


    