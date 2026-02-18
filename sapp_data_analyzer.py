#!/usr/bin/env python3

"""

SAPP Market Data Analyzer Tool

Interactive Python script for analyzing Southern African Power Pool energy market data


Usage: python sapp_data_analyzer.py

"""


import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')


class SAPPAnalyzer:

    def __init__(self, filepath):

        """Initialize the analyzer with the Excel file"""

        print("Loading SAPP Market Data...")

        self.df = pd.read_excel(filepath, sheet_name='Market data', skiprows=1)

        

        # Set proper column names

        self.df.columns = ['Date', 'Year', 'Month', 'Hour', 'ToU_Period', 

                          'DAM_Price', 'DAM_Volume', 

                          'Monthly_Price', 'Monthly_Volume',

                          'Weekly_Price', 'Weekly_Volume', 

                          'IntraDay_Price', 'IntraDay_Volume']

        

        # Convert data types

        self.df['Date'] = pd.to_datetime(self.df['Date'])

        self.df['Hour'] = pd.to_numeric(self.df['Hour'], errors='coerce')

        

        # Convert prices and volumes to numeric

        price_cols = ['DAM_Price', 'Monthly_Price', 'Weekly_Price', 'IntraDay_Price']

        volume_cols = ['DAM_Volume', 'Monthly_Volume', 'Weekly_Volume', 'IntraDay_Volume']

        

        for col in price_cols + volume_cols:

            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        

        print(f"✓ Loaded {len(self.df):,} records from {self.df['Date'].min()} to {self.df['Date'].max()}")

    

    def summary_statistics(self):

        """Display summary statistics"""

        print("\n" + "="*70)

        print("SUMMARY STATISTICS")

        print("="*70)

        

        print("\nPrice Statistics ($/MWh):")

        print(self.df[['DAM_Price', 'Monthly_Price', 'Weekly_Price', 'IntraDay_Price']].describe())

        

        print("\nVolume Statistics (MWh):")

        print(self.df[['DAM_Volume', 'Monthly_Volume', 'Weekly_Volume', 'IntraDay_Volume']].describe())

    

    def filter_by_date(self, start_date, end_date):

        """Filter data by date range"""

        mask = (self.df['Date'] >= start_date) & (self.df['Date'] <= end_date)

        return self.df[mask]

    

    def filter_by_year(self, year):

        """Filter data by year"""

        return self.df[self.df['Year'] == year]

    

    def get_peak_hours(self, market='DAM_Price'):

        """Get peak price hours"""

        hourly_avg = self.df.groupby('Hour')[market].mean().sort_values(ascending=False)

        return hourly_avg.head(5)

    

    def compare_markets(self):

        """Compare different market types"""

        comparison = pd.DataFrame({

            'Average Price ($/MWh)': [

                self.df['DAM_Price'].mean(),

                self.df['Monthly_Price'].mean(),

                self.df['Weekly_Price'].mean(),

                self.df['IntraDay_Price'].mean()

            ],

            'Total Volume (MWh)': [

                self.df['DAM_Volume'].sum(),

                self.df['Monthly_Volume'].sum(),

                self.df['Weekly_Volume'].sum(),

                self.df['IntraDay_Volume'].sum()

            ],

            'Price Volatility (Std)': [

                self.df['DAM_Price'].std(),

                self.df['Monthly_Price'].std(),

                self.df['Weekly_Price'].std(),

                self.df['IntraDay_Price'].std()

            ]

        }, index=['DAM', 'Monthly', 'Weekly', 'IntraDay'])

        

        return comparison

    

    def plot_price_trends(self, market='DAM', save_path=None):

        """Plot price trends over time"""

        fig, ax = plt.subplots(figsize=(14, 6))

        

        # Monthly averages

        monthly_data = self.df.groupby(self.df['Date'].dt.to_period('M'))[f'{market}_Price'].mean()

        monthly_data.index = monthly_data.index.to_timestamp()

        

        ax.plot(monthly_data.index, monthly_data.values, linewidth=2, color='steelblue')

        ax.fill_between(monthly_data.index, monthly_data.values, alpha=0.3, color='steelblue')

        

        ax.set_title(f'{market} Price Trends Over Time (Monthly Average)', fontsize=14, fontweight='bold')

        ax.set_xlabel('Date', fontsize=12)

        ax.set_ylabel('Price ($/MWh)', fontsize=12)

        ax.grid(True, alpha=0.3)

        

        plt.tight_layout()

        

        if save_path:

            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            print(f"✓ Chart saved to {save_path}")

        else:

            plt.show()

    

    def plot_hourly_patterns(self, save_path=None):

        """Plot hourly price patterns"""

        fig, ax = plt.subplots(figsize=(12, 6))

        

        hourly_avg = self.df.groupby('Hour')['DAM_Price'].mean()

        

        ax.bar(hourly_avg.index, hourly_avg.values, color='coral', alpha=0.7)

        ax.set_title('Average DAM Price by Hour of Day', fontsize=14, fontweight='bold')

        ax.set_xlabel('Hour', fontsize=12)

        ax.set_ylabel('Average Price ($/MWh)', fontsize=12)

        ax.grid(True, alpha=0.3, axis='y')

        

        plt.tight_layout()

        

        if save_path:

            plt.savefig(save_path, dpi=300, bbox_inches='tight')

            print(f"✓ Chart saved to {save_path}")

        else:

            plt.show()

    

    def export_summary_csv(self, filepath):

        """Export monthly summary to CSV"""

        monthly_summary = self.df.groupby(self.df['Date'].dt.to_period('M')).agg({

            'DAM_Price': ['mean', 'max', 'min', 'std'],

            'DAM_Volume': 'sum',

            'Monthly_Price': 'mean',

            'Weekly_Price': 'mean',

            'IntraDay_Price': 'mean'

        }).round(2)

        

        monthly_summary.to_csv(filepath)

        print(f"✓ Monthly summary exported to {filepath}")

    

    def get_price_extremes(self, market='DAM_Price', n=10):

        """Get highest and lowest price records"""

        highest = self.df.nlargest(n, market)[['Date', 'Hour', market, f'{market.split("_")[0]}_Volume']]

        lowest = self.df[self.df[market] > 0].nsmallest(n, market)[['Date', 'Hour', market, f'{market.split("_")[0]}_Volume']]

        

        return {'Highest': highest, 'Lowest': lowest}




# ==============================================================================

# INTERACTIVE MENU

# ==============================================================================


def main_menu():

    """Display interactive menu"""

    print("\n" + "="*70)

    print("        SAPP MARKET DATA ANALYZER - INTERACTIVE MENU")

    print("="*70)

    print("\n1.  View Summary Statistics")

    print("2.  Compare All Markets")

    print("3.  Plot Price Trends (by Market)")

    print("4.  Plot Hourly Patterns")

    print("5.  Get Peak Price Hours")

    print("6.  Get Price Extremes (Highest/Lowest)")

    print("7.  Filter by Year")

    print("8.  Export Monthly Summary to CSV")

    print("9.  View Data Sample")

    print("0.  Exit")

    print("\n" + "="*70)

    

    return input("\nSelect an option (0-9): ")




def main():

    """Main function to run the analyzer"""

    print("="*70)

    print("     SAPP MARKET DATA ANALYZER")

    print("     Southern African Power Pool - Energy Trading Analysis")

    print("="*70)

    

    # Initialize analyzer with your file path

    filepath = input("\nEnter path to Excel file (or press Enter for default): ").strip()

    if not filepath:
        filepath = 'Data Analyst SAPP Market Extract.xlsx'  # <-- Change this line

    

    try:

        analyzer = SAPPAnalyzer(filepath)

    except Exception as e:

        print(f"\n✗ Error loading file: {e}")

        return

    

    while True:

        choice = main_menu()

        

        if choice == '1':

            analyzer.summary_statistics()

            

        elif choice == '2':

            print("\nMARKET COMPARISON:")

            print(analyzer.compare_markets())

            

        elif choice == '3':

            market = input("\nEnter market type (DAM/Monthly/Weekly/IntraDay): ").strip()

            save = input("Save chart? (y/n): ").strip().lower()

            save_path = f"{market}_price_trends.png" if save == 'y' else None

            analyzer.plot_price_trends(market, save_path)

            

        elif choice == '4':

            save = input("\nSave chart? (y/n): ").strip().lower()

            save_path = "hourly_patterns.png" if save == 'y' else None

            analyzer.plot_hourly_patterns(save_path)

            

        elif choice == '5':

            market = input("\nEnter market type (DAM_Price/Monthly_Price/Weekly_Price/IntraDay_Price): ").strip()

            print("\nTop 5 Peak Price Hours:")

            print(analyzer.get_peak_hours(market))

            

        elif choice == '6':

            market = input("\nEnter market type (DAM_Price/Monthly_Price/Weekly_Price/IntraDay_Price): ").strip()

            n = int(input("How many records to show? (default 10): ") or 10)

            extremes = analyzer.get_price_extremes(market, n)

            print(f"\n{n} Highest Prices:")

            print(extremes['Highest'])

            print(f"\n{n} Lowest Prices (excluding zero):")

            print(extremes['Lowest'])

            

        elif choice == '7':

            year = int(input("\nEnter year (2017-2022): "))

            year_data = analyzer.filter_by_year(year)

            print(f"\nData for {year}:")

            print(f"Total records: {len(year_data):,}")

            print(f"Average DAM Price: ${year_data['DAM_Price'].mean():.2f}/MWh")

            print(f"Total DAM Volume: {year_data['DAM_Volume'].sum():,.0f} MWh")

            

        elif choice == '8':

            filename = input("\nEnter filename (default: monthly_summary.csv): ").strip()

            if not filename:

                filename = 'monthly_summary.csv'

            analyzer.export_summary_csv(filename)

            

        elif choice == '9':

            n = int(input("\nHow many rows to display? (default 10): ") or 10)

            print("\nData Sample:")

            print(analyzer.df.head(n))

            

        elif choice == '0':

            print("\nExiting analyzer. Goodbye!")

            break

            

        else:

            print("\n✗ Invalid option. Please try again.")

        

        input("\nPress Enter to continue...")




if __name__ == "__main__":

    main()
