import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import scipy.interpolate as interp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import yfinance as yf
from scipy.optimize import minimize
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BlackScholes:
    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def d2(S, K, T, r, sigma):
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S, K, T, r, sigma):
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
    
    @staticmethod
    def put_price(S, K, T, r, sigma):
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)
        return K * np.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)
    
    @staticmethod
    def implied_volatility(price, S, K, T, r, option_type='call'):
        if price <= 0 or T <= 0:
            return 0.0
            
        def objective(sigma):
            if option_type.lower() == 'call':
                return abs(BlackScholes.call_price(S, K, T, r, sigma) - price)
            else:
                return abs(BlackScholes.put_price(S, K, T, r, sigma) - price)
        
        try:
            result = minimize(objective, 0.2, method='L-BFGS-B', bounds=[(0.001, 5.0)])
            return result.x[0]
        except:
            return 0.0

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x**2)

class DataFetcher:
    def __init__(self):
        self.ticker = "SPY"
        
    def fetch_risk_free_rate(self):
        try:
            irx = yf.Ticker("^IRX")
            irx_data = irx.history(period="1d")
            if not irx_data.empty:
                rate = irx_data['Close'].iloc[-1] / 100.0
                return rate
            return 0.04
        except Exception as e:
            logger.error(f"Error fetching risk-free rate: {str(e)}")
            return 0.04
    
    def fetch_options_data(self):
        try:
            start_time = time.time()
            logger.info(f"Downloading options data for {self.ticker}")
            
            ticker = yf.Ticker(self.ticker)
            stock_price = ticker.history(period="1d")['Close'].iloc[-1]
            risk_free_rate = self.fetch_risk_free_rate()
            
            expiration_dates = ticker.options
            
            options_data = []
            current_time = dt.datetime.now()
            
            for expiry_str in expiration_dates:
                try:
                    expiry_date = dt.datetime.strptime(expiry_str, '%Y-%m-%d')
                    days_to_expiry = (expiry_date - current_time).days
                    
                    if days_to_expiry <= 0:
                        continue
                        
                    T = days_to_expiry / 365.0
                    
                    option_chain = ticker.option_chain(expiry_str)
                    calls = option_chain.calls
                    puts = option_chain.puts
                    
                    for _, call in calls.iterrows():
                        try:
                            strike = float(call['strike'])
                            market_price = float(call['lastPrice'])
                            volume = int(call['volume']) if not np.isnan(call['volume']) else 0
                            open_interest = int(call['openInterest']) if not np.isnan(call['openInterest']) else 0
                            
                            if volume < 5 and open_interest < 10:
                                continue
                            
                            if np.isnan(call['impliedVolatility']):
                                implied_vol = BlackScholes.implied_volatility(
                                    market_price, stock_price, strike, T, risk_free_rate, 'call'
                                )
                            else:
                                implied_vol = call['impliedVolatility']
                            
                            if implied_vol <= 0.01 or implied_vol > 2.0:
                                continue
                                
                            options_data.append({
                                "timestamp": current_time,
                                "expiration": expiry_date,
                                "days_to_expiry": days_to_expiry,
                                "strike": strike,
                                "call_put": "call",
                                "implied_volatility": implied_vol,
                                "price": market_price,
                                "volume": volume,
                                "open_interest": open_interest
                            })
                        except Exception as e:
                            continue
                    
                    for _, put in puts.iterrows():
                        try:
                            strike = float(put['strike'])
                            market_price = float(put['lastPrice'])
                            volume = int(put['volume']) if not np.isnan(put['volume']) else 0
                            open_interest = int(put['openInterest']) if not np.isnan(put['openInterest']) else 0
                            
                            if volume < 5 and open_interest < 10:
                                continue
                            
                            if np.isnan(put['impliedVolatility']):
                                implied_vol = BlackScholes.implied_volatility(
                                    market_price, stock_price, strike, T, risk_free_rate, 'put'
                                )
                            else:
                                implied_vol = put['impliedVolatility']
                            
                            if implied_vol <= 0.01 or implied_vol > 2.0:
                                continue
                                
                            options_data.append({
                                "timestamp": current_time,
                                "expiration": expiry_date,
                                "days_to_expiry": days_to_expiry,
                                "strike": strike,
                                "call_put": "put",
                                "implied_volatility": implied_vol,
                                "price": market_price,
                                "volume": volume,
                                "open_interest": open_interest
                            })
                        except Exception as e:
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing expiry {expiry_str}: {str(e)}")
                    continue
            
            elapsed_time = time.time() - start_time
            logger.info(f"Downloaded data for {len(options_data)} options in {elapsed_time:.2f} seconds")
            
            return {
                "options_data": options_data,
                "stock_price": stock_price,
                "risk_free_rate": risk_free_rate
            }
            
        except Exception as e:
            logger.error(f"Error downloading options data: {str(e)}")
            raise

class VolatilitySurfaceCalculator:
    def __init__(self, data_fetcher):
        self.data_fetcher = data_fetcher
    
    def calculate_volatility_surface(self):
        try:
            data_result = self.data_fetcher.fetch_options_data()
            options_data = data_result["options_data"]
            stock_price = data_result["stock_price"]
            
            call_options = [opt for opt in options_data if opt["call_put"] == "call"]
            
            if len(call_options) < 10:
                raise Exception("Insufficient data to calculate volatility surface")
            
            strikes = sorted(list(set([opt["strike"] for opt in call_options])))
            days_list = sorted(list(set([opt["days_to_expiry"] for opt in call_options])))
            
            strike_coords = []
            days_coords = []
            iv_values = []
            
            for option in call_options:
                strike_coords.append(option["strike"])
                days_coords.append(option["days_to_expiry"])
                iv_values.append(option["implied_volatility"])
            
            strike_grid = np.linspace(min(strikes), max(strikes), 50)
            days_grid = np.linspace(min(days_list), max(days_list), 50)
            strike_mesh, days_mesh = np.meshgrid(strike_grid, days_grid)
            
            points = np.vstack((strike_coords, days_coords)).T
            iv_surface = interp.griddata(
                points, iv_values, (strike_mesh, days_mesh), method='cubic', fill_value=np.nan
            )
            
            mask = np.isnan(iv_surface)
            if np.any(mask):
                iv_surface_nearest = interp.griddata(
                    points, iv_values, (strike_mesh, days_mesh), method='nearest'
                )
                iv_surface[mask] = iv_surface_nearest[mask]
            
            current_time = dt.datetime.now()
            surface_data = {
                "strike_mesh": strike_mesh.tolist(),
                "days_mesh": days_mesh.tolist(),
                "iv_surface": iv_surface.tolist(),
                "timestamp": current_time.isoformat(),
                "stock_price": stock_price,
                "raw_options": call_options
            }
            
            return surface_data
        except Exception as e:
            logger.error(f"Error calculating volatility surface: {str(e)}")
            raise

class VolatilitySurfaceVisualizer:
    @staticmethod
    def create_3d_vol_surface(surface_data):
        strike_mesh = np.array(surface_data["strike_mesh"])
        days_mesh = np.array(surface_data["days_mesh"])
        iv_surface = np.array(surface_data["iv_surface"])
        stock_price = surface_data.get("stock_price", 0)
        
        fig = go.Figure(data=[
            go.Surface(
                x=strike_mesh,
                y=days_mesh,
                z=iv_surface * 100,
                colorscale='Viridis',
                colorbar=dict(title="IV (%)")
            )
        ])
        
        if stock_price > 0:
            y_min = np.min(days_mesh)
            y_max = np.max(days_mesh)
            
            fig.add_trace(go.Scatter3d(
                x=[stock_price, stock_price],
                y=[y_min, y_max],
                z=[0, np.max(iv_surface) * 100 * 1.2],
                mode='lines',
                line=dict(color='red', width=5),
                name='Current price'
            ))
        
        fig.update_layout(
            title=f"SPY Implied Volatility Surface",
            scene=dict(
                xaxis_title="Strike Price ($)",
                yaxis_title="Days to Expiration",
                zaxis_title="Implied Volatility (%)",
                xaxis=dict(gridcolor="lightgray"),
                yaxis=dict(gridcolor="lightgray"),
                zaxis=dict(gridcolor="lightgray")
            ),
            autosize=True,
            height=700,
        )
        
        return fig
    
    @staticmethod
    def create_vol_smile_plots(surface_data):
        options = surface_data["raw_options"]
        stock_price = surface_data.get("stock_price", 0)
        
        expirations = {}
        for option in options:
            days = option["days_to_expiry"]
            if days not in expirations:
                expirations[days] = []
            expirations[days].append(option)
        
        days_list = sorted(expirations.keys())
        
        if len(days_list) > 5:
            selected_days = [days_list[0]]
            for i in range(1, 4):
                idx = int(i * len(days_list) / 4)
                selected_days.append(days_list[idx])
            selected_days.append(days_list[-1])
        else:
            selected_days = days_list
        
        fig = make_subplots(rows=1, cols=len(selected_days), 
                            subplot_titles=[f"{int(days)} Days" for days in selected_days])
        
        for i, days in enumerate(selected_days):
            options_by_strike = sorted(expirations[days], key=lambda x: x["strike"])
            
            strikes = [opt["strike"] for opt in options_by_strike]
            ivs = [opt["implied_volatility"] * 100 for opt in options_by_strike]
            
            fig.add_trace(
                go.Scatter(
                    x=strikes,
                    y=ivs,
                    mode='lines+markers',
                    name=f'{int(days)} days'
                ),
                row=1, col=i+1
            )
            
            if stock_price > 0:
                fig.add_shape(
                    type="line",
                    x0=stock_price, y0=0,
                    x1=stock_price, y1=max(ivs) * 1.1,
                    line=dict(color="red", width=1, dash="dash"),
                    row=1, col=i+1
                )
        
        fig.update_layout(
            title="Volatility Smiles by Expiration",
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Strike Price ($)")
        fig.update_yaxes(title_text="Implied Volatility (%)")
        
        return fig
    
    @staticmethod
    def create_term_structure_plot(surface_data):
        options = surface_data["raw_options"]
        stock_price = surface_data.get("stock_price", 0)
        
        all_strikes = sorted(set(opt["strike"] for opt in options))
        
        if stock_price > 0:
            target_strikes = [
                stock_price * 0.9, 
                stock_price * 0.95,
                stock_price,
                stock_price * 1.05,
                stock_price * 1.1
            ]
            
            selected_strikes = []
            for target in target_strikes:
                closest = min(all_strikes, key=lambda x: abs(x - target))
                if closest not in selected_strikes:
                    selected_strikes.append(closest)
        else:
            if len(all_strikes) > 5:
                step = len(all_strikes) // 5
                selected_strikes = [all_strikes[0]]
                for i in range(1, 4):
                    selected_strikes.append(all_strikes[i * step])
                selected_strikes.append(all_strikes[-1])
            else:
                selected_strikes = all_strikes
        
        fig = go.Figure()
        
        atm_strike = min(all_strikes, key=lambda x: abs(x - stock_price)) if stock_price > 0 else 0
        
        for strike in selected_strikes:
            strike_options = [opt for opt in options if abs(opt["strike"] - strike) < 0.01]
            
            strike_options.sort(key=lambda x: x["days_to_expiry"])
            
            if strike_options:
                days = [opt["days_to_expiry"] for opt in strike_options]
                ivs = [opt["implied_volatility"] * 100 for opt in strike_options]
                
                line_style = "solid"
                line_width = 2
                
                if abs(strike - atm_strike) < 0.01:
                    line_width = 4
                
                if stock_price > 0:
                    moneyness = (strike / stock_price - 1) * 100
                    label = f'${strike:.0f} ({moneyness:+.1f}%)'
                else:
                    label = f'${strike:.0f}'
                
                fig.add_trace(go.Scatter(
                    x=days,
                    y=ivs,
                    mode='lines+markers',
                    name=label,
                    line=dict(width=line_width, dash=line_style),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title="Volatility Term Structure by Strike",
            xaxis_title="Days to Expiration",
            yaxis_title="Implied Volatility (%)",
            height=600,
            legend_title="Strike Price"
        )
        
        return fig
    
    @staticmethod
    def create_skew_analysis_plot(surface_data):
        options = surface_data["raw_options"]
        stock_price = surface_data.get("stock_price", 0)
        
        expirations = {}
        for option in options:
            days = option["days_to_expiry"]
            if days not in expirations:
                expirations[days] = []
            expirations[days].append(option)
        
        skew_data = []
        
        for days, exp_options in expirations.items():
            exp_options.sort(key=lambda x: x["strike"])
            
            if len(exp_options) < 3:
                continue
                
            if stock_price > 0:
                atm_idx = min(range(len(exp_options)), key=lambda i: abs(exp_options[i]["strike"] - stock_price))
            else:
                atm_idx = len(exp_options) // 2
                
            atm_strike = exp_options[atm_idx]["strike"]
            atm_iv = exp_options[atm_idx]["implied_volatility"]
            
            put_strike_target = atm_strike * 0.95
            put_idx = min(range(len(exp_options)), key=lambda i: abs(exp_options[i]["strike"] - put_strike_target))
            put_iv = exp_options[put_idx]["implied_volatility"]
            
            skew = put_iv - atm_iv
            
            skew_data.append({
                "days": days,
                "atm_iv": atm_iv * 100,
                "put_iv": put_iv * 100,
                "skew": skew * 100
            })
        
        skew_data.sort(key=lambda x: x["days"])
        
        if not skew_data:
            return go.Figure()
            
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        days = [d["days"] for d in skew_data]
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=[d["skew"] for d in skew_data],
                mode='lines+markers',
                name='Skew (25d Put - ATM)',
                line=dict(color='red', width=2)
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=days,
                y=[d["atm_iv"] for d in skew_data],
                mode='lines+markers',
                name='ATM IV',
                line=dict(color='blue', width=2)
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Volatility Skew Analysis by Expiration",
            xaxis_title="Days to Expiration",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            height=500,
        )
        
        fig.update_yaxes(title_text="Skew (25d Put - ATM) (%)", secondary_y=False)
        fig.update_yaxes(title_text="ATM IV (%)", secondary_y=True)
        
        return fig
        
    @staticmethod
    def create_stats_table(surface_data):
        options = surface_data["raw_options"]
        stock_price = surface_data.get("stock_price", 0)
        
        expirations = {}
        for option in options:
            days = option["days_to_expiry"]
            if days not in expirations:
                expirations[days] = []
            expirations[days].append(option)
        
        days_list = sorted(expirations.keys())
        
        stats_data = []
        
        for days in days_list:
            exp_options = expirations[days]
            
            exp_options.sort(key=lambda x: x["strike"])
            
            if stock_price > 0:
                atm_idx = min(range(len(exp_options)), key=lambda i: abs(exp_options[i]["strike"] - stock_price))
            else:
                atm_idx = len(exp_options) // 2
                
            atm_strike = exp_options[atm_idx]["strike"]
            atm_iv = exp_options[atm_idx]["implied_volatility"] * 100
            
            otm_put_target = atm_strike * 0.95
            otm_call_target = atm_strike * 1.05
            
            put_idx = min(range(len(exp_options)), key=lambda i: abs(exp_options[i]["strike"] - otm_put_target))
            call_idx = min(range(len(exp_options)), key=lambda i: abs(exp_options[i]["strike"] - otm_call_target))
            
            otm_put_iv = exp_options[put_idx]["implied_volatility"] * 100
            otm_call_iv = exp_options[call_idx]["implied_volatility"] * 100
            
            put_skew = otm_put_iv - atm_iv
            call_skew = otm_call_iv - atm_iv
            
            min_iv = min(opt["implied_volatility"] for opt in exp_options) * 100
            max_iv = max(opt["implied_volatility"] for opt in exp_options) * 100
            
            stats_data.append({
                "days": int(days),
                "date": (dt.datetime.now() + dt.timedelta(days=int(days))).strftime("%m/%d/%Y"),
                "atm_iv": atm_iv,
                "min_iv": min_iv,
                "max_iv": max_iv,
                "put_skew": put_skew,
                "call_skew": call_skew,
                "skew_ratio": put_skew / call_skew if call_skew != 0 else 0
            })
        
        if stats_data:
            df = pd.DataFrame(stats_data)
            return df
        
        return pd.DataFrame()

def streamlit_app():
    st.set_page_config(
        page_title="SPY 3D Volatility Surface",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("SPY Options - 3D Volatility Surface")
    
    st.sidebar.title("Controls")
    
    refresh_button = st.sidebar.button("Refresh Data")
    
    tab_options = ["3D Surface", "Volatility Smile", "Term Structure", "Skew Analysis", "Statistics"]
    selected_tab = st.sidebar.radio("Select View", tab_options)
    
    if refresh_button or 'surface_data' not in st.session_state:
        with st.spinner("Downloading and calculating volatility surface..."):
            try:
                data_fetcher = DataFetcher()
                calculator = VolatilitySurfaceCalculator(data_fetcher)
                st.session_state.surface_data = calculator.calculate_volatility_surface()
                st.sidebar.success("Data updated successfully!")
            except Exception as e:
                st.sidebar.error(f"Error updating data: {str(e)}")
                if 'surface_data' not in st.session_state:
                    st.session_state.surface_data = None
    
    if selected_tab == "3D Surface":
        st.header("3D Volatility Surface")
        
        if 'surface_data' in st.session_state and st.session_state.surface_data:
            stock_price = st.session_state.surface_data.get("stock_price", 0)
            if stock_price > 0:
                st.write(f"Current SPY price: ${stock_price:.2f}")
                
            fig = VolatilitySurfaceVisualizer.create_3d_vol_surface(st.session_state.surface_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("What does this visualization show?"):
                st.write("The 3D volatility surface shows how implied volatility (z-axis) varies with strike price (x-axis) and time to expiration (y-axis). Key features include the volatility skew (higher IV for lower strikes), smile pattern, and term structure. The red line indicates the current SPY price.")
        else:
            st.error("No data available. Try refreshing the data.")
    
    elif selected_tab == "Volatility Smile":
        st.header("Volatility Smile")
        
        if 'surface_data' in st.session_state and st.session_state.surface_data:
            fig = VolatilitySurfaceVisualizer.create_vol_smile_plots(st.session_state.surface_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to interpret the volatility smile?"):
                st.write("Each graph shows the volatility smile or skew for a specific expiration. The dashed red line indicates the current price (at-the-money). Options to the left are OTM puts/ITM calls, while options to the right are ITM puts/OTM calls. In a perfect Black-Scholes market, this curve would be flat. The skew (higher on the left side) reflects demand for downside protection.")
        else:
            st.error("No data available. Try refreshing the data.")
    
    elif selected_tab == "Term Structure":
        st.header("Volatility Term Structure")
        
        if 'surface_data' in st.session_state and st.session_state.surface_data:
            fig = VolatilitySurfaceVisualizer.create_term_structure_plot(st.session_state.surface_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to interpret the term structure?"):
                st.write("This chart shows how implied volatility varies with expiration for different strike levels. Each line represents a different strike (shown in the legend with % moneyness). The thicker line represents the at-the-money strike. An upward sloping curve indicates contango (higher volatility for longer expirations), while downward sloping indicates backwardation.")
        else:
            st.error("No data available. Try refreshing the data.")
    
    elif selected_tab == "Skew Analysis":
        st.header("Volatility Skew Analysis")
        
        if 'surface_data' in st.session_state and st.session_state.surface_data:
            fig = VolatilitySurfaceVisualizer.create_skew_analysis_plot(st.session_state.surface_data)
            st.plotly_chart(fig, use_container_width=True)
            
            with st.expander("How to interpret the skew analysis?"):
                st.write("This chart shows two key metrics: Skew (red line, left axis) representing the difference between OTM put and ATM volatility for each expiration, and ATM IV (blue line, right axis) showing at-the-money implied volatility. A high positive skew indicates the market is pricing significant downside risk. Sudden changes in skew can signal shifts in market sentiment.")
        else:
            st.error("No data available. Try refreshing the data.")
            
    elif selected_tab == "Statistics":
        st.header("Volatility Statistics by Expiration")
        
        if 'surface_data' in st.session_state and st.session_state.surface_data:
            stats_df = VolatilitySurfaceVisualizer.create_stats_table(st.session_state.surface_data)
            
            if not stats_df.empty:
                formatted_df = stats_df.copy()
                formatted_df['atm_iv'] = formatted_df['atm_iv'].map('{:.2f}%'.format)
                formatted_df['min_iv'] = formatted_df['min_iv'].map('{:.2f}%'.format)
                formatted_df['max_iv'] = formatted_df['max_iv'].map('{:.2f}%'.format)
                formatted_df['put_skew'] = formatted_df['put_skew'].map('{:+.2f}%'.format)
                formatted_df['call_skew'] = formatted_df['call_skew'].map('{:+.2f}%'.format)
                formatted_df['skew_ratio'] = formatted_df['skew_ratio'].map('{:.2f}'.format)
                
                formatted_df.columns = [
                    'Days', 'Expiration Date', 'ATM IV', 'Min IV', 'Max IV', 
                    'Put Skew', 'Call Skew', 'Skew Ratio'
                ]
                
                st.dataframe(formatted_df)
                
                st.subheader("Key Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                atm_iv_30d = stats_df[stats_df['days'] <= 30]['atm_iv'].mean()
                col1.metric("30-day Average IV", f"{atm_iv_30d:.2f}%")
                
                put_skew_30d = stats_df[stats_df['days'] <= 30]['put_skew'].mean()
                col2.metric("30-day Avg Put Skew", f"{put_skew_30d:+.2f}%")
                
                short_term = stats_df[stats_df['days'] <= 30]['atm_iv'].mean()
                long_term = stats_df[stats_df['days'] >= 60]['atm_iv'].mean()
                term_slope = long_term - short_term
                col3.metric("Term Structure Slope", f"{term_slope:+.2f}%")
                
                with st.expander("How to interpret these statistics?"):
                    st.write("This table provides detailed volatility statistics for each expiration. ATM IV shows at-the-money implied volatility. Put/Call Skew shows the difference between OTM and ATM volatility. Min/Max IV provides range. The key metrics summarize current volatility conditions: 30-day Average IV shows short-term volatility, Average Put Skew indicates skew steepness, and Term Structure Slope shows the difference between long and short-term volatility (positive = contango, negative = backwardation).")
            else:
                st.warning("Insufficient statistical data for analysis.")
        else:
            st.error("No data available. Try refreshing the data.")

if __name__ == "__main__":
    streamlit_app()
