from modern_portfolio_theory import ModernPortfolioTheory 
import streamlit as st
import requests

st.title("Modern Portfolio Theory - Portfolio Optimizer (2014-2024 data)")

key_array = []

col1, col2 = st.columns(2)


with col1:
    key_array.append(st.text_input('Symbol 1'))
    key_array.append(st.text_input('Symbol 2'))
    key_array.append(st.text_input('Symbol 3'))
    key_array.append(st.text_input('Symbol 4'))
    key_array.append(st.text_input('Symbol 5'))

value_array = []

with col2:
    value_array.append(st.text_input('Avanza ID for Security 1'))
    value_array.append(st.text_input('Avanza ID for Security 2'))
    value_array.append(st.text_input('Avanza ID for Security 3'))
    value_array.append(st.text_input('Avanza ID for Security 4'))
    value_array.append(st.text_input('Avanza ID for Security 5'))

rf=float(st.text_input('Risk-free interest rate %', value=1))/100

# Create the portfolio dictionary only if both key and value are filled
portfolio = {k: v for k, v in zip(key_array, value_array) if k and v}

# Check if portfolio data is valid before proceeding
if portfolio:
    portfolio_data = ModernPortfolioTheory(portfolio)
    
    # Display the optimal allocation
    optimal_allocation, fig_optimal = portfolio_data.get_best_allocation(rf)
    
   
    for key in portfolio.keys():
        optimal_allocation[key] = optimal_allocation[key] * 100
        optimal_allocation = optimal_allocation.rename(columns={key: f"{key.title()} (%)"})

    
    optimal_allocation = optimal_allocation.rename(columns={key: f"{key.title()} (%)"})

    optimal_allocation["std"]*=100
    optimal_allocation=optimal_allocation.rename(columns={"std":"Annual volatility (std %)"})

    optimal_allocation["ret"]*=100
    optimal_allocation=optimal_allocation.rename(columns={"ret":"Annual return (mean %)"})

        

    st.dataframe(optimal_allocation,hide_index=True)
    
    
    # Display the first plot (efficient frontier)
    st.pyplot(fig_optimal)
    
    # Display the second plot (comparison with index)
    fig_compare,compare_df = portfolio_data.compare_to_index(rf)  # This returns the figure object
    
    st.dataframe(compare_df)
    st.pyplot(fig_compare)  # Display the second figure



else:
    st.write("Please enter valid Avanza IDs (see below)")
    st.image(requests.get('https://drive.google.com/uc?export=view&id=1NDLfHw-SFziYcLLelO_wGc_456_PlttU').content)
