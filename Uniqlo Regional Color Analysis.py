# UNIQLO REGIONAL COLOR ANALYSIS 

# SECTION 1: IMPORT LIBRARIES AND SETUP
import pandas as pd
import numpy as np                    # For the mathematical operations and calculations
import matplotlib.pyplot as plt
import seaborn as sns                 # For creating beautiful statistical visualizations
from scipy import stats               # For statistical tests (t-tests, chi-square)
import warnings                       # For controlling warning messages
warnings.filterwarnings("ignore")     # Hide minor warning messages for cleaner output


# Set display options for better data viewing
pd.set_option("display.max_columns", None)   # Show all columns when printing datasets
pd.set_option("display.width", 1000)         # Make the display wider for better readability

print("=== UNIQLO REGIONAL COLOR ANALYSIS ===")
print("STEP 1: Loading and Exploring Datasets\n")

# SECTION 2: LOAD THE DATASETS
# The "try-except" block handles errors gracefully
# 1. If files load successfully, continue with the analysis.
# 2. If files are missing, show helpful error message.

try:
    products_df = pd.read_csv("Products.csv")
    stores_df = pd.read_csv("Store.csv")
    sales_df = pd.read_csv("Sales.csv")

    print("✅ All datasets loaded successfully")
    print(f"Products dataset: {products_df.shape}")   # Shows (rows, columns)
    print(f"Stores dataset: {stores_df.shape}")
    print(f"Sales dataset: {sales_df.shape}")

except FileNotFoundError as e:
    # If any CSV file is missing, this error message appears.
    print(f"❌ File not found: {e}")
    print("Please ensure your CSV files are in the correct directory")


# SECTION 3: EXPLORE THE DATASETS
# This section shows you what's inside each dataset - like opening each Excel sheet to see the data.

print("\n" + "="*50)
print("DATA STRUCTURES")
print("="*50)

print("\n📊 PRODUCTS DATASET")
print(products_df.head())
print(f"\nColumns: {list(products_df.columns)}")   # lists all the column names
print(f"Data Types:\n{products_df.dtypes}")        # Shows the type of the columns

print("\n🏪 STORES DATASET")
print(stores_df.head())
print(f"\nColumns: {list(stores_df.columns)}")
print(f"Data Types:\n{stores_df.dtypes}")

print("\n💰 SALES DATASET")
print(sales_df.head())
print(f"\nColumns: {list(sales_df.columns)}")
print(f"Data Types:\n{sales_df.dtypes}")

# SECTION 4: DATA QUALITY CHECK
# This section checks for common data problems that could affect your analysis.

print("\n" + "="*50)
print("DATA QUALITY CHECK")
print("="*50)

# This function checks each dataset for potential issues:
def data_quality_check(df, name):
    print(f"\n{name}:")
    print(f" - Total rows: {len(df)}")                          # How many records in the dataset
    print(f" - Missing values: {df.isnull().sum().sum()}")      # Empty cells that need attention
    print(f" - Duplicate rows: {df.duplicated().sum()}")          # Repeated records that could skew results

    # If there are missing values. show which columns are the problem:
    if df.isnull().sum().sum() > 0:
        print(f" - Columns with missing values: {df.columns[df.isnull().any()].tolist()}")

# Run the quality check on each dataset:
data_quality_check(products_df, "Products")
data_quality_check(stores_df, "Stores")
data_quality_check(sales_df, "Sales")

# SECTION 5: SUMMARY AND NEXT STEPS
# This section summarizes what we accomplished and whats coming next

print("\n" + "="*50)
print("NEXT STEPS")
print("="*50)
print("1. ✅ Data loaded and structure verified")
print("2. 🔄 Next: Data merging and derived metrics creation")
print("3. 📈 Then: Exploratory data analysis")
print("4. 🎯 Finally: Core research questions analysis")


# STEP 2: DATA MERGING AND DERIVED METRICS
# This step combines all datasets and creates key metrics for color analysis

print("\n" + "="*60)
print("STEP 2: DATA MERGING AND DERIVED METRICS CREATION")
print("="*60)

# SECTION 1: DATA CLEANING AND PREPARATION

# Clean up store dataset - remove empty column
if "Unnamed: 7" in stores_df.columns:
    stores_df = stores_df.drop("Unnamed: 7", axis = 1)
    print("✅ Cleaned stores dataset - removed empty column")

# Convert date column to datetime for better analysis
sales_df["Sale_Date"] = pd.to_datetime(sales_df["Sale_Date"])
print("✅ Converted sales dates to datetime format")

# Fix price column in products (remove "$" and convert to float)
products_df["Price_Numeric"] = products_df["Price"].str.replace("$", "")
print("✅ Created numeric price column")

# SECTION 2: CREATE MASTER DATASET
print("\n📊 CREATING MASTER DATASET")

# Step 1: Merge sales with products (to get product details for each sale)
sales_products = sales_df.merge(
    products_df,
    on="Product_ID",
    how="left",
    suffixes=("_sale", "_product")   # Handles duplicate column names
)

# Step 2: Merge with stores (to get regional information for each sale)
master_df = sales_products.merge(
    stores_df,
    on="Store_ID",
    how="left"
)

print(f"✅ Master dataset created: {master_df.shape[0]} rows, {master_df.shape[1]} columns")

# SECTION 3: DATA VALIDATION
print("\n🔍 VALIDATING MERGED DATA...")

# Check for successful merges
unmatched_products = master_df["Product_Name"].isnull().sum()
unmatched_stores = master_df["Store_name"].isnull().sum()

if unmatched_products == 0 and unmatched_stores == 0:
    print("✅ All records successfully merged")
else:
    print("⚠️ Some records didnt match")


# SECTION 4: CREATE KEY METRICS FOR ANALYSIS
print("\n📈 CREATING ANALYSIS METRICS...")

# Add month and year for seasonal analysis
master_df["Sale_Month"] = master_df["Sale_Date"].dt.month
master_df["Sale_Year"] = master_df["Sale_Date"].dt.year
master_df["Month_Name"] = master_df["Sale_Date"].dt.strftime("%B")

print("All columns:", list(master_df.columns))

# Look for color related columns
color_cols = [col for col in master_df.columns if "color" in col.lower() or "colour" in col.lower()]
print("Color related columns found:", color_cols)

# Check if the expected column exists
expected_cols = ["Color_sale", "Color", "Colour", "Color_product", "Colour_product"]
exisiting_expected = [col for col in expected_cols if col in master_df.columns]
print("Expected color columns that exist:", exisiting_expected)

# Create a unified color column based on what actually exists
if "Colour_sale" in master_df.columns and "Color" in master_df.columns:
    master_df["Color_Unified"] = master_df["Colour_sale"].fillna(master_df["Color"])
    print("✅ Used Colour_sale and Color columns")
elif "Color_sale" in master_df.columns and "Color" in master_df.columns:
    master_df["Color_Unified"] = master_df["Color_sale"].fillna(master_df["Color"])
    print("✅ Used Color_sale and Color columns")
elif "Colour" in master_df.columns:
    master_df["Color_Unified"] = master_df["Colour"]
    print("✅ Used Colour column only")
elif "Color" in master_df.columns:
    master_df["Color_Unified"] = master_df["Color"]
    print("✅ Used Color column only")
elif len(color_cols) > 0:
    master_df["Color_Unified"] = master_df[color_cols[0]]
    print(f"✅ Used first available color column: {color_cols[0]}")
else:
    master_df["Color_Unified"] = "Unknown"
    print("⚠️ No color columns found, using Unknown")


# Add revenue per unit metric
if "Total_Revenue" in master_df.columns and "Qty_Sold" in master_df.columns:
    master_df["Revenue_per_unit"] = master_df["Total_Revenue"] / master_df["Qty_Sold"]
    print("✅ Created revenue per unit metric")
else:
    print("⚠️ Could not create revenue per unit metric.")

# Create Regional groupings for easier analysis
region_mapping = {
    "Cool": ["Stockholm", "Rotterdam"],
    "Cool_Neutral": ["Paris", "Toronto"],
    "Neutral": ["Luxembourg City", "New York"],
    "Neutral_Warm": ["Tokyo", "Beijing"],
    "Warm": ["Mumbai", "Taipei"]
}

# Add region cateogry
def assign_region_category(City):
    if pd.isna(City):
        return "Other"
    for region, cities in region_mapping.items():
        if City in cities:
            return region
    return "Other"

if "City" in master_df.columns:
    master_df["Region_Category"] = master_df["City"].apply(assign_region_category)
    print("✅ Created regional category groupings")
else:
    print("⚠️ Could not create regional categories")
    master_df["Region_Cateogry"] = "Other"

print("✅ Created key analysis metrics:")
print(" - Sale month/year for seasonal analysis")
print(" - Unified color column")
print(" - Revenue per unit calculation")
print(" - Regional category groupings")

# SECTION 5: DATA SET OVERVIEW
print("\n" + "="*50)
print("MASTER DATASET OVERVIEW")
print("="*50)

print(f"📊 Dataset Dimensions: {master_df.shape}")
print(f"📅 Data Range: {master_df['Sale_Date'].min()} to {master_df['Sale_Date'].max()}")
print(f"🏪 Stores Covered: {master_df['Store_ID'].nunique()}")
print(f"🛍️ Products Covered: {master_df['Product_ID'].nunique()}")
print(f"🎨 Colours Available: {master_df['Color_Unified'].nunique()}")
print(f"🌍 Cities Covered: {master_df['City'].nunique()}")

# Show sample of master dataset
print("\n SAMPLE DATA:")
print(master_df[["Sales_ID", "Product_Name", "Color_Unified", "City", "Dominant_Undertone", "Region_Category", "Qty_Sold", "Total_Revenue"]].head())

# Show regional distribution
print("\n🌍 SALES BY REGION CATEOGRY:")
region_sales = master_df.groupby("Region_Category").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count"
}).round(2)
region_sales.columns = ["Total_Qty_Sold", "Total_Revenue", "Number_Of_Sales"]
print(region_sales)

print("\n" + "="*50)
print("NEXT STEPS")
print("="*50)
print("✅ STEP 2 COMPLETE: Master dataset ready for analysis")
print("🔄 Next: Step 3 - Exploratory Data Analysis")
print("🎯 Then: Step 4 - Core Research Questions Analysis")
print("📊 Finally: Results preparation for Tableau visualization")

# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# This step explores patterns in your data to understand color preferences across regions.

print("\n" + "="*60)
print("STEP 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

# SECTION 1: OVERALL DATA DISTRIBUTION
print("\n📊 SECTION 1: OVERALL DATA DISTRIBUTION")
print("="*50)

# 1.1 Sales Volume by Region
print("\n🌍 SALES VOLUME BY REGION:")
region_summary = master_df.groupby("Region_Category").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count"
}).round(2)

region_summary.columns = ["Total_Qty_Sold", "Total_Revenue", "Number_of_Sales"]

region_summary["Avg_Qty_Per_Sale"] = (region_summary["Total_Qty_Sold"] / region_summary["Number_of_Sales"]).round(2)
region_summary["Avg_Revenue_Per_Sale"] = (region_summary["Total_Revenue"] / region_summary["Number_of_Sales"]).round(2)
print(region_summary)

# 1.2 Product Category Performance
print("\n🛍️ PRODUCT CATEGORY PERFORMANCE")
category_performance = master_df.groupby("Category").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count"
}).round(2)

category_performance.columns = ["Total_Qty_Sold", "Total_Revenue", "Number_of_Sales"]
category_summary = category_performance.sort_values("Total_Revenue", ascending=False)
print(category_summary)

# 1.3 Subcategory Performance
print("\n👕 SUBCATEGORY PERFORMANCE:")
subcategory_perfomance = master_df.groupby("Subcategory").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).round(2)

subcategory_perfomance.columns = ["Total_Qty_Sold", "Total_Revenue"]
subcategory_summary = subcategory_perfomance.sort_values("Total_Revenue", ascending=False)
print(subcategory_summary)

# SECTION 2: COLOR ANALYSIS
print("\n" + "="*50)
print("\n📊 COLOR ANALYSIS")
print("="*50)

# 2.1 Overall Color Performance
print("\n🎨 TOP PERFORMING COLORS (By Revenue)")
color_performance = master_df.groupby("Color_Unified").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count"
}).round(2)

color_performance = color_performance.sort_values("Total_Revenue", ascending=False)
print(color_performance.head(15))

# 2.2 Color Performance by Region
print("\n🌍 COLOR PERFORMANCE BY REGION:")
print("All Columns:", list(master_df.columns))
print("Region-related columns:", [col for col in master_df.columns if "region" in col.lower() or "category" in col.lower()])

color_by_region = master_df.groupby(["Region_Category", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).round(2)

print("\nTop 3 colors by revenue in each region:")
for region in master_df["Region_Category"].unique():
    if region != "Other":
        region_data = color_by_region.loc[region].sort_values("Total_Revenue", ascending=False)
        print(f"\n{region}:")
        print(region_data.head(3))
    

# 2.3 Color Performance by Product Category
print("\n" + "="*50)
print("🛍️ COLOR PERFORMANCE BY PRODUCT CATEGORY")
print("="*50)

for category in master_df["Category"].unique():
    print(f"\n {category.upper()}:")
    category_colors = master_df[master_df["Category"] == category].groupby("Color_Unified").agg({
        "Qty_Sold": "sum",
        "Total_Revenue": "sum"
    }).sort_values("Total_Revenue", ascending=False)
    print(category_colors.head(5))

# SECTION 3: SEASONAL ANALYSIS
print("\n" + "="*50)
print("📊 SECTION 3: SEASONAL ANAYLSIS")
print("="*50)

# 3.1 Monthly Sales Trends
print("\n📅 MONTHLY SALES TRENDS")
monthly_sales = master_df.groupby("Month_Name").agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).round(2)

# Reorder by month
month_order = ["January", "Feburary", "March", "April", "May", "June",
                 "July", "August", "September", "October", "November", "December"]

monthly_sales = monthly_sales.reindex([m for m in month_order if m in monthly_sales.index])
print(monthly_sales)

# 3.2 Seasonal Color Preferences
print("\n🎨 SEASONAL COLOR PREFERENCES:")

# Map the months to the seasons
def get_season(month):
    if month in ["October", "November", "December", "January", "Feburary", "March"]:
        return "Fall/Winter"
    else:
        return "Spring/Summer"
    
master_df["Seasonal_Actual"] = master_df["Month_Name"].apply(get_season)

seasonal_colors = master_df.groupby(["Seasonal_Actual", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).round(2)

print("\nTop 5 colors by revenue in each season:")
for season in ["Spring/Summer", "Fall/Winter"]:
    if season in seasonal_colors.index.get_level_values(0):
        season_data = seasonal_colors.loc[season].sort_values("Total_Revenue", ascending=False)
        print(f"\n{season}:")
        print(season_data.head(5))

# SECTION 4: REGIONAL SKIN TONE ANALYSIS
print("\n" + "="*50)
print("📊 REGIONAL SKIN TONE ANALYSIS")
print("="*50)

# 4.1 Skin Tone Distribution by Region
print("\n🌍 SKIN TONE DISTRIBUTION BY REGION:")
undertone_by_region = master_df.groupby(["Region_Category", "Dominant_Undertone"]).agg({
    "Sales_ID": "count",
    "Total_Revenue": "sum"
}).round(2)
print(undertone_by_region)

# 4.2 Color Performance by Skin Tone
print("\n🎨 COLOR PERFORMANCE BY SKIN TONE")
for undertone in master_df["Dominant_Undertone"].unique():
    if pd.notna(undertone):
        print(f"\n{undertone} Undertone - Top Colors:")
        undertone_colors = master_df[master_df["Dominant_Undertone"] == undertone].groupby("Color_Unified").agg({
            "Qty_Sold": "sum",
            "Total_Revenue": "sum"
        }).sort_values("Total_Revenue", ascending=False)
        print(undertone_colors.head(5))

# SECTION 5: PRICE ANALYSIS
print("\n" + "="*50)
print("📊 SECTION 5: PRICE ANALYSIS")
print("="*50)

# 5.1 Price Distribution by Color
print("\n💰 AVERAGE PRICE BY COLOR:")

# First, convert Price_Numeric to float if its still string.
if master_df["Price_Numeric"].dtype == "object":
    master_df["Price_Numeric"] = pd.to_numeric(master_df["Price_Numeric"], errors = "coerce")

price_by_color = master_df.groupby("Color_Unified")["Price_Numeric"].agg(["mean","count"]).round(2)
price_by_color.columns = ["Avg_Price", "Produce_Count"]
price_by_color = price_by_color.sort_values("Avg_Price", ascending=False)
print(price_by_color.head(15))

# 5.2 Revenue per unit by region
print("\n💵 REVENUE PER UNIT BY REGION")
revenue_per_unit_region = master_df.groupby("Region_Category")["Revenue_per_unit"].agg(["mean", "median"]).round(2)
print(revenue_per_unit_region)

# SECTION 6: KEY INSIGHTS SUMMARY
print("\n" + "="*60)
print("🎯 KEY INSIGHTS FROM EDA")
print("="*60)

# Calculate some key metrics for insights
top_region = region_summary["Total_Revenue"].idxmax()
top_color = color_performance["Total_Revenue"].idxmax()
top_category = category_summary["Total_Revenue"].idxmax()
top_month = monthly_sales["Total_Revenue"].idxmax()

print(f"🏆 PERFORMANCE LEADERS:")
print(f" - Top Revenue Region: {top_region}")
print(f" - Top Selling Color: {top_color}")
print(f" - Top Product Category: {top_category}")
print(f" - Peak Sales Month: {top_month}")

print(f"\n📊 SCALE INSIGHTS:")
print(f" - Total Colors Available: {master_df['Color_Unified'].nunique()}")
print(f" - Average Colors per Region: {master_df.groupby('Region_Category')['Color_Unified'].nunique().mean():.1f}")
print(f" - Most Color Variety Region: {master_df.groupby('Region_Category')['Color_Unified'].nunique().idxmax()}")

print(f"\n💡 READY FOR RESEARCH QUESTIONS:")
print("   • Color-Region alignment analysis")
print("   • Product-specific color performance")
print("   • Seasonal color optimization opportunities")
print("   • Regional inventory optimization potential")

print("\n" + "="*50)
print("NEXT STEPS")
print("="*50)
print("✅ STEP 3 COMPLETE: Exploratory analysis reveals key patterns")
print("🔄 Next: Step 4 - Core Research Questions Analysis")
print("📈 Then: Statistical testing and recommendations")
print("📊 Finally: Results preparation for Tableau visualization")

# STEP 4: CORE RESEARCH QUESTIONS ANALYSIS
# This step will answer the main business questions of the analysis.

print("\n" + "="*70)
print("STEP 4: CORE RESEARCH QUESTIONS ANALYSIS")
print("="*70)

# RESEARCH QUESTION 1: Which product styles perform best in specific colors?
print("\n" + "="*60)
print("🎯 RESEARCH QUESTION 1:")
print("Which product styles perform best in specific colors?")
print("="*60)

# 1.1 Product-Color Performance Matrix
print("\n📊 SECTION 1.1 - PRODUCT-COLOR PERFORMANCE ANALYSIS")
print("-"*50)

# Create comprehensive product-color performance table
product_color_performance = master_df.groupby(["Subcategory", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count",
    "Revenue_per_unit": "mean"
}).round(2)

product_color_performance.columns = ["Total_Qty", "Total_Revenue", "Sales_Count", "Avg_Revenue_Per_Unit"]

# Calculate performance metrics
product_color_performance["Revenue_per_Sale"] = (product_color_performance["Total_Revenue"] / product_color_performance["Sales_Count"]).round(2)

print("\n🏆 TOP 20 PRODUCT-COLOR COMBINATIONS (By Total Revenue:)")
top_combinations = product_color_performance.sort_values("Total_Revenue", ascending=False)
print(top_combinations.head(20))

# 1.2 Best Colors for each product category
print("\n📦 BEST COLORS FOR EACH PRODUCT CATEGORY")
print("-"*50)

for subcategory in master_df["Subcategory"].unique():
    print(f"\n {subcategory.upper()}:")

    # Get performance for this subcategory
    subcategory_data = product_color_performance.loc[subcategory].sort_values("Total_Revenue", ascending=False)

    # Show top 5 colors
    print(" Top 5 Colors by Revenue:")
    for i, (color, row) in enumerate(subcategory_data.head(5).iterrows(), 1):
        print(f" {i}. {color}: ${row['Total_Revenue']:,.2f} revenue, {row['Total_Qty']} units")


# 1.3 Statistical Significances Testing
print("\n📈 SECTION 1.3 - STATISTICAL ANALYSIS")
print("-"*50)

from scipy.stats import chi2_contingency, f_oneway
print("\n🔬 STATISTICAL SIGNIFICANCES TESTING")

# Create contingency table for chi-square test
contingency_data = master_df.groupby(["Subcategory", "Color_Unified"])["Qty_Sold"].sum().unstack(fill_value=0)

print("\n✅ Chi-Square Test Results:")
print("Testing if color preferences vary significantly across product categories")

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_data)
print(f"Chi-Square statistic: {chi2:.4f}")
print(f"P-Value: {p_value:.6f}")
print(f"Degrees of Freedom: {dof}")

if p_value > 0.05:
    print("🎯 RESULT: Color Preferences DO vary significantly across product categories")
else:
    print("📊 RESULT: Color Preferences DO NOT vary signicantly across product categories")

# RESEARCH QUESTION 2: Regional Color Optimization
print("\n" + "="*60)
print("🎯 RESEARCH QUESTION 2:")
print("Can we optimize color assortment by aligning with regional skin tone preferences?")
print("="*60)

# 2.1 Regional Color Performance Analysis
print("\n🌍 SECTION 2.1 - REGIONAL COLOR PERFORMANCE ANALYSIS")
print("-"*50)

# Analyze color performance by region
regional_color_analysis = master_df.groupby(["Region_Category", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count"
}).round(2)

print("\n🏆 TOP 5 COLORS BY REGION (Revenue-based):")
for region in master_df["Region_Category"].unique():
    if region != "Other":
        print(f"\n🌎 {region.replace('_','-')} Region:")

        region_data = regional_color_analysis.loc[region].sort_values("Total_Revenue", ascending=False)

        for i, (color,row) in enumerate(region_data.head(5).iterrows(), 1):
            market_share = (row["Total_Revenue"] / region_data["Total_Revenue"].sum()*100)
            print(f" {i}. {color}: ${row['Total_Revenue']:,.2f} ({market_share:.1f}% of regional revenue)")

# 2.2 Skin Tone VS Color Performance Analysis
print("\n🎨 SECTION 2.2 - SKIN TONE ALIGNMENT ANALYSIS")
print("-"*50)

# Analyze how well current color choices align with skin tone theory
skin_tone_analysis = master_df.groupby(["Dominant_Undertone", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).round(2)

print("\n COLOR PREFERENCES BY SKIN UNDERTONE")
for undertone in master_df["Dominant_Undertone"].unique():
    if pd.notna(undertone):
        print(f"\n {undertone} Undertone:")

        undertone_data = skin_tone_analysis.loc[undertone].sort_values("Total_Revenue", ascending=False)

        total_undertone_revenue = undertone_data["Total_Revenue"].sum()
        for i, (color,row) in enumerate(undertone_data.head(5).iterrows(), 1):
            percentage = (row["Total_Revenue"] / total_undertone_revenue * 100)
            print(f" {i}. {color}: ${row['Total_Revenue']:,.2f} ({percentage:.1f}%)")

# 2.3 Color Theory Alignment Score
print("\n🎯 SECTION 2.3 - COLOR THEORY ALIGNMENT SCORE")
print("-"*50)

# Define theorectical color preferences based on undertones
# Function to map actual Uniqlo colors to color theory categoryies

def get_color_theory_category(uniqlo_color):
    """
    Maps Uniqlo's specific color codes to color theory categories.
    """

    color = str(uniqlo_color).upper().strip()

    # Cool undertones (blue/pink undertones work best)
    if color in ['09 BLACK', '00 WHITE', '69 NAVY', '60 LIGHT BLUE', '61 BLUE', '62 BLUE', 
                 '63 BLUE', '65 BLUE', '67 BLUE', '68 BLUE', '10 PINK', '11 PINK', '12 PINK',
                 '70 LIGHT PURPLE', '72 PURPLE', '76 PURPLE', '03 GRAY', '07 GRAY']:
        return 'Cool'
    
    # Warm undertones (yellow/golden undertones work best)
    elif color in ['31 BEIGE', '32 BEIGE', '38 DARK BROWN', '21 LIGHT ORANGE', '15 RED',
                   '30 NATURAL', '56 OLIVE', '57 OLIVE']:
        return 'Warm'
    
    # Cool-Neutral (can handle some cool and some neutral colors)
    elif color in ['54 GREEN', '58 DARK GREEN', '59 DARK GREEN', '19 WINE']:
        return 'Cool_Neutral'
    
    # Neutral-Warm (slight warm undertones)
    elif color in ['01 OFF WHITE', '02 LIGHT GRAY', '08 DARK GRAY']:
        return 'Neutral_Warm'
    
    # Default to Neutral (balanced undertones)
    else:
        return 'Neutral'

# Apply Color Theory categorization to the dataset
print("🎨 MAPPING UNIQLO COLORS TO THE COLOR THEORY CATEGORIES:")
master_df["Color_Theory_Category"] = master_df["Color_Unified"].apply(get_color_theory_category) 

# Show mapping results
print("\n📊 COLOR THEORY CATEGORY DISTRIBUTION:")
color_theory_distribution = master_df["Color_Theory_Category"].value_counts()
for category, count in color_theory_distribution.items():
    percentage = (count / len(master_df) * 100)
    print(f" {category.replace('_','-')}: {count:,} sales ({percentage:.1f}%)")

# Show which specific Uniqlo colors fall into each theory category
print("\n🎨 UNIQLO COLORS BY THORY CATEGORY")
color_mapping_summary = master_df.groupby(["Color_Theory_Category", "Color_Unified"]).size().reset_index(name="Count")

for category in ["Cool", "Cool_Neutral", "Neutral", "Neutral_Warm", "Warm"]:
    if category in color_mapping_summary["Color_Theory_Category"].values:
        colors_in_category = color_mapping_summary[color_mapping_summary["Color_Theory_Category"] == category]
        print(f"\n{category.replace('_','-')} ({len(colors_in_category)} colors):")
        for _, row in colors_in_category.iterrows():
            print(f" - {row['Color_Unified']}")

print("\n📊 THEORECTICAL VS ACTUAL COLOR PERFORMANCE:")

alignment_scores = {}

print("\n🎨 CREATING DYNAMIC COLOR THEORY MAPPING")
print("-"*50)

# Get all unique colors in your dataset
all_colors = master_df['Color_Unified'].unique()
print(f"Found {len(all_colors)} unique colors in dataset:")
print(list(all_colors))

# Create color theory mapping based on the color theory categories you already defined
color_theory_mapping = {}

for region in ['Cool', 'Cool_Neutral', 'Neutral', 'Neutral_Warm', 'Warm']:
    # Get colors that theoretically match this region's undertone
    region_colors = []
    
    for color in all_colors:
        color_category = get_color_theory_category(color)
        
        # Match colors to regions based on theory
        if region == 'Cool' and color_category in ['Cool']:
            region_colors.append(color)
        elif region == 'Cool_Neutral' and color_category in ['Cool', 'Cool_Neutral', 'Neutral']:
            region_colors.append(color)
        elif region == 'Neutral' and color_category in ['Neutral', 'Cool_Neutral', 'Neutral_Warm']:
            region_colors.append(color)
        elif region == 'Neutral_Warm' and color_category in ['Neutral_Warm', 'Neutral', 'Warm']:
            region_colors.append(color)
        elif region == 'Warm' and color_category in ['Warm', 'Neutral_Warm']:
            region_colors.append(color)
    
    # If no perfect matches, include some neutral colors as backup
    if len(region_colors) < 5:
        neutral_colors = [c for c in all_colors if get_color_theory_category(c) == 'Neutral']
        region_colors.extend(neutral_colors[:5-len(region_colors)])
    
    color_theory_mapping[region] = region_colors[:10]  # Top 10 theoretical colors

# Display the dynamic mapping
print("\n📋 DYNAMIC COLOR THEORY MAPPING:")
for region, colors in color_theory_mapping.items():
    print(f"\n{region.replace('_', '-')} Region:")
    print(f"  Theoretical colors ({len(colors)}): {colors}")

print(f"\n✅ Dynamic color theory mapping created for {len(color_theory_mapping)} regions")

# Validation: Check if mapping has colors for all regions
for region in master_df['Region_Category'].unique():
    if region != 'Other' and region in color_theory_mapping:
        if len(color_theory_mapping[region]) == 0:
            print(f"⚠️ Warning: No colors mapped for {region}")
        else:
            print(f"✅ {region}: {len(color_theory_mapping[region])} colors mapped")

for region in master_df["Region_Category"].unique():
    if region != "Other" and region in color_theory_mapping:

        # Get actual top colors in this region
        region_colors = regional_color_analysis.loc[region].sort_values("Total_Revenue", ascending=False)
        actual_top_5 = region_colors.head(5).index.tolist()

        # Get theorectical best colors
        theorectical_colors = color_theory_mapping[region]

        # Calculate alignment score
        matches = sum(1 for color in actual_top_5 if any(theo_color.lower() in color.lower() for theo_color in theorectical_colors))
        alignment_score = (matches / 5) * 100

        alignment_scores[region]  = alignment_score

        print(f"\n🎨 {region.replace('_','-')} Region:")
        print(f" Theorectical Best Colors: {theorectical_colors[:5]}")
        print(f" Actual Top 5 Colors: {actual_top_5}")
        print(f" Alignment Score: {alignment_score:.1f}% ({matches}/5 matches)")

# 2.4 Optimization Opportunities
print("\n💡 OPTIMIZATION OPPORTUNITIES")
print("-"*50)

print("\n🎯 REGIONAL OPTIMIZATION RECOMMENDATIONS:")

for region, score in alignment_scores.items():
    if score < 60:  # Low Alignment 
        theorectical_colors = color_theory_mapping[region]
        region_data = regional_color_analysis.loc[region].sort_values("Total_Revenue", ascending=False)

        print(f"\n {region.replace('_','-')} Region (Alignment: {score:.1f}%):")
        print(" 📈 RECOMMENDED ACTIONS:")
        print(f" Increase Inventory of: {', '.join(theorectical_colors[:3])}")

        # Find Underperforming Theorectical Colors
        available_colors = master_df[master_df["Region_Category"] == region]["Color_Unified"].unique()
        missing_oppurtunities = []

        for theo_color in theorectical_colors[:5]:
            matching_colors = [c for c in available_colors if theo_color.lower() in c.lower()]
            if matching_colors:
                color_performance = region_data.loc[matching_colors[0]] if matching_colors[0] in region_data.index else None
                if color_performance is not None:
                    rank = list(region_data.index).index(matching_colors[0]) + 1
                    if rank > 5:
                        missing_oppurtunities.append(f"{matching_colors[0]} (currently rank #{rank})")
        
        if missing_oppurtunities:
            print(f" Focus on marketing on: {', '.join(missing_oppurtunities[:2])}")


print("\n" + "="*60)
print("🎯 REGIONAL COLOR EXPANSION OPPORTUNITIES (COLOR THEORY-BASED)")
print("="*60)

for region in alignment_scores.keys():
    if region == "Other":
        continue
    theory_colors = color_theory_mapping[region]
    # Only consider colors that exist in your master product set (the 33)
    all_available_colors = set(master_df["Color_Unified"].unique())
    available_in_region = set(master_df[master_df["Region_Category"] == region]["Color_Unified"].unique())
    # Colors that are theory recommended and in the master set, but not in this region
    missing_theory_colors = [color for color in theory_colors if color in all_available_colors and color not in available_in_region]

    print(f"\nRegion: {region}")
    print(f"  Theory-recommended colors for this region: {theory_colors}")
    print(f"  Currently sold in this region: {sorted(available_in_region)}")
    print(f"  🚀 NEW COLOR OPPORTUNITIES (not currently sold, recommended by theory): {missing_theory_colors if missing_theory_colors else 'None'}")

    if missing_theory_colors:
        # Estimate potential revenue uplift for adding these colors, if desired
        avg_revenue = master_df[master_df["Region_Category"] == region].groupby("Color_Unified")["Total_Revenue"].mean().mean()
        potential_uplift = avg_revenue * len(missing_theory_colors)
        print(f"     (Estimated revenue uplift if added: ${potential_uplift:,.2f})")
    else:
        print("  ✅ All theory-recommended colors are already present in this region.")


# SECTION 3: BUSINESS IMPACT ANALYSIS
print("\n" + "="*60)
print("💰 SECTION 3: BUSINESS IMPACT AND RECOMMENDATIONS")
print("="*60)

# 3.1 Revenue Impact Calculation
print("\n📈 POTENTIAL REVENUE IMPACT:")

total_revenue = master_df["Total_Revenue"].sum()
print(f"Current Total Revenue: ${total_revenue:,.2f}")

# Calculate potential uplift from better alignment 
average_alignment = np.mean(list(alignment_scores.values()))
potential_uplift = (80 - average_alignment) / 100 * 0.15          # Assume 15% uploft potential per 100% alignment improvemnt

potential_additional_revenue = total_revenue * potential_uplift
print(f"Current Regional Alignment: {average_alignment:.1f}%")
print(f"Potential Revenue Uplift: ${potential_additional_revenue:,.2f} ({potential_uplift*100:.1f}%)")

# 3.2 Product-Specific Recommendations
print("\n🛍️ PRODUCT-SPECIFIC RECOMMENDATIONS")

# Find best product-color combinations that could expand
for subcategory in ["T-shirts", "Jeans", "Dresses", "Jackets", "Sweaters"]:
    if subcategory in product_color_performance.index:
        subcat_data = product_color_performance.loc[subcategory].sort_values("Total_Revenue", ascending=False)
        top_color = subcat_data.index[0]
        top_revenue = subcat_data.iloc[0]["Total_Revenue"]

        print(f"\n {subcategory}:")
        print(f"   Best performing color: {top_color} (${top_revenue:,.2f} revenue)")

        # Check if this top color is available in all regions
        color_regions = master_df[
            (master_df["Subcategory"] == subcategory) &
            (master_df["Color_Unified"] == top_color)]["Region_Category"].unique()
        
        total_regions = len([r for r in master_df["Region_Category"].unique() if r != "Other"])

        if len(color_regions) < total_regions:
            missing_regions = set(master_df["Region_Category"].unique()) - set(color_regions) - {"Other"}
            print(f" 📊 Expansison Opportunity: Available in {len(color_regions)} / {total_regions} regions")
            if missing_regions:
                print(f" 🎯 Consider introducing to: {', '.join(missing_regions)}")

print("\n" + "="*60)
print("🎯 KEY FINDINGS SUMMARY")
print("="*60)

print(f"\n📊 RESEARCH QUESTION 1 FINDINGS:")
print(f" - Identified {len(top_combinations)} product-color combinations")
print(f" - Color Preferences {'DO' if p_value < 0.05 else 'DO NOT'} vary significantly across categories")
print(f" - Statistical Significance: p = {p_value:.6f}")

print(f"\n🌍 RESEARCH QUESTION 2 FINDINGS:")
print(f" - Average regional alignment with color theory: {average_alignment:.1f}%")
print(f" - Regions with optimization opportunities: {sum(1 for score in alignment_scores.values() if score < 60)}")
print(f" - Potential revenue impact: ${potential_additional_revenue:,.2f}")

print(f"\n💡 NEXT STEPS:")
print("   ✅ Core analysis complete - ready for Tableau visualization")
print("   📊 Statistical insights validated")
print("   🎯 Business recommendations generated")
print("   📈 Revenue impact quantified")

# STEP 5: TABLEAU PREPARATION AND DATA EXPORT 

print("\n" + "="*50)
print("STEP 5: TABLEAU PREPARATION AND DATA EXPORT")
print("="*50)

# SECTION 1: CREATE TABLEAU-READY DATASETS
print("\n📊 SECTION 1: CREATE TABLEAU-READY DATASETS:")
print("-"*50)

# 1.1 Preparing Master Dataset
print("\n🔧 Preparing the Master Dataset:")

tableau_master_dataset = master_df.copy()

# Add calculated fields that tableau will use
tableau_master_dataset['Year'] = tableau_master_dataset['Sale_Date'].dt.year
tableau_master_dataset['Quarter'] = tableau_master_dataset['Sale_Date'].dt.quarter

try:
    tableau_master_dataset['Week'] = tableau_master_dataset['Sale_Date'].dt.isocalendar().week
except AttributeError:
    tableau_master_dataset['Week'] = tableau_master_dataset['Sale_Date'].dt.week

# Create performance metrics
tableau_master_dataset['Revenue_Rank_by_Region'] = tableau_master_dataset.groupby('Region_Category')['Total_Revenue'].rank(ascending=False)
tableau_master_dataset['Revenue_Rank_by_Color'] = tableau_master_dataset.groupby('Color_Unified')['Total_Revenue'].rank(ascending=False)

print(f"✅ Master dataset prepared: {tableau_master_dataset.shape}")

# Clean and standardize data for tableau
tableau_master_dataset = tableau_master_dataset.rename(columns={
    "Region_Category": "Region_Undertone",
    "Color_Unified": "Color",
    "Qty_Sold": "Quantity_Sold",
    "Sale_Date": "Date",
    "Month_Name": "Month",
    "Dominant_Undertone": "Skin_Undertone"
})


# 1.2 Product-Color Performance Summary
print("\n📦 Preapring Product-Color Performance Summary")

product_color_summary = master_df.groupby(["Subcategory", "Color_Unified", "Region_Category"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count",
    "Revenue_per_unit": "mean"
}).reset_index()

product_color_summary.columns = ["Product_Type", "Color", "Region", "Total_Quantity", "Total_Revenue", "Sales_Count", "Avg_Revenue_Per_Unit"]

# Add performance rankings
product_color_summary["Revenue_Rank"] = product_color_summary.groupby("Product_Type")["Total_Revenue"].rank(ascending=False)
product_color_summary["Quantity_Rank"] = product_color_summary.groupby("Product_Type")["Total_Quantity"].rank(ascending=False)

print("\n✅ Product-Color Summary Created: {product_color_summary.shape}")

# 1.3 Regional Performance Summary 
print("\n🌍 Preparing Regional Performance Dataset:")

regional_summary = master_df.groupby(["Region_Category", "Dominant_Undertone", "City", "Country"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum",
    "Sales_ID": "count",
    "Color_Unified": "nunique"
}).reset_index()

regional_summary.columns = ["Region_Undertone", "Skin_Undertone", "City", "Country", "Total_Quantity", "Total_Revenue", "Sales_Count", "Color_Variety"]

# Add market share calculations
total_revenue = region_summary["Total_Revenue"].sum()
regional_summary["Revenue_Market_Share"] = (regional_summary["Total_Revenue"] / total_revenue * 100).round(2)

print(f"✅ Regional summary created: {regional_summary.shape}")

# 1.4 Color Theory Alignment Dataset
print("\n🎨 Preparing Color Theory Alignment Dataset:")

# RESEARCH BASED Color theory mapping for Tableau
# Based on professional seasonal color analysis principles.

color_theory_data = []
uniqlo_dataset_colors = master_df["Color_Unified"].unique()

# Create alignment dataset using the uniqlo colors and existing color theory categories
for region in ["Cool", "Cool_Neutral", "Neutral", "Neutral_Warm", "Warm"]:
    for color in uniqlo_dataset_colors:
        color_category = get_color_theory_category(color)

        # Determine if this color is theoretically good for this region
        is_recommended = False
        if region == "Cool" and color_category == "Cool":
            is_recommended = True
        elif region == "Cool_Neutral" and color_category in ["Cool_Neutral", "Cool", "Neutral"]:
            is_recommended = True
        elif region == "Neutral" and color_category in ["Cool_Neutral", "Neutral", "Neutral_Warm"]:
            is_recommended = True
        elif region == "Neutral_Warm" and color_category in ["Neutral_Warm", "Warm", "Neutral"]:
            is_recommended = True
        elif region == "Warm" and color_category in ["Warm", "Neutral_Warm"]:
            is_recommended = True
        
        color_theory_data.append({
            "Region_Undertone": region,
            "Color": color,
            "Color_Theory_Category": color_category,
            "Theoretical_Recommendation": "Recommend" if is_recommended else "Not Optimal",
            "Recommendation_Score": 1 if is_recommended else 0 
        })

color_theory_df = pd.DataFrame(color_theory_data)
print(f"✅ Color Theory dataset created: {color_theory_df.shape}")

# 1.5 Monthly Trends Dataset
print("\n📅 Preparing Monthly Trends Dataset:")

monthly_trends = master_df.groupby(["Sale_Month", "Month_Name", "Region_Category", "Color_Unified"]).agg({
    "Qty_Sold": "sum",
    "Total_Revenue": "sum"
}).reset_index()

monthly_trends.columns = ["Month_Number", "Month_Name", "Region", "Color", "Quantity_Sold", "Revenue"]

# Add seasonal grouping
def assign_season(month):
    if month in [9,10,11,12,1,2]:
        return "Fall/Winter"
    else:
        return "Spring/Summer"

monthly_trends["Season"] = monthly_trends["Month_Number"].apply(assign_season)
print(f"✅ Monthly Trends Dataset created: {monthly_trends.shape}")

# 1.6 Alignment Performance Dataset
print("\n🎯 Preparing Alignment Performance Dataset:")

# Create a dataset that shows how well each region aligns with the color theory
alignment_performance = []

for region in master_df["Region_Category"].unique():
    if region != "Other":

        # Get actual sales performance in this region
        region_data = master_df[master_df["Region_Category"] == region]
        region_color_performance = region_data.groupby("Color_Unified").agg({
            "Qty_Sold": "sum",
            "Total_Revenue": "sum"
        }).reset_index()

        # Get top 5 colors by revenue
        top_colors = region_color_performance.nlargest(5, "Total_Revenue")["Color_Unified"].tolist()

        # Calculate alignment with color theory
        theory_colors = color_theory_df[
            (color_theory_df["Region_Undertone"] == region) &
            (color_theory_df["Theoretical_Recommendation"] == "Recommend")
        ]["Color"].tolist()
        
        # Calculate alignment metrics
        alignment_count = sum(1 for color in top_colors if color in theory_colors)
        alignment_percentage = (alignment_count / 5) * 100 if len(top_colors) >= 5 else 0

        alignment_performance.append({
            "Region": region,
            "Top_5_Actual_Colors": ", ".join(top_colors),
            "Theoretical_Colors_Count": len(theory_colors),
            "Alignment_Count": alignment_count,
            "Alignment_Percentage": alignment_percentage,
            "Total_Revenue": region_color_performance["Total_Revenue"].sum(),
            "Total_Quantity": region_color_performance["Qty_Sold"].sum()
        })

alignment_df = pd.DataFrame(alignment_performance)
print(f"✅ Alignment Performance Dataset created: {alignment_df.shape}")

# SECTION 2: EXPORT DATASETS FOR TABLEAU
print("\n💾 SECTION 2: EXPORTING DATASETS")
print("-" * 50)

# Create export directory
import os
export_dir = "Tableau_Data"
if not os.path.exists(export_dir):
    os.makedirs(export_dir)
    print(f"📁 CREATED DIRECTORY: {export_dir}")

# Export all datasets
datasets_to_export = {
    "Master_Dataset": tableau_master_dataset,
    "Product_Color_Performance": product_color_summary,
    "Regional_Performance": regional_summary,
    "Color_Theory_Alignment": color_theory_df,
    "Montly_Trends": monthly_trends,
    "Alignment_Performance": alignment_df
}

print("\n📤 EXPORTING DATASETS")
for name, dataset in datasets_to_export.items():
    filepath = os.path.join(export_dir, f"{name}.csv")
    dataset.to_csv(filepath, index=False)
    print(f"✅ EXPORTED: {name}.csv ({dataset.shape[0]} rows, {dataset.shape[1]} columns)")

# SECTION 3: TABLEAU DASHBOARD SPECIFICATIONS
print("\n📊 SECTION 3: TABLEAU DASHBOARD SPECIFICATIONS")
print("-" * 50)

# Show sample of key datasets for verification
print("\n🔍 SAMPLE DATA VERIFICATION:")

print("\n📋 Color Theory Alignment Sample:")
print(color_theory_df.head(10))

print("\n📋 Alignment Performance Sample:")
print(alignment_df)

print("\n📋 Regional Performance Sample:")
print(regional_summary.head())

tableau_specs = """
🎯 RECOMMENDED TABLEAU DASHBOARDS:

1. 📈 EXECUTIVE OVERVIEW DASHBOARD
   - KPI Cards: Total Revenue, Units Sold, Regions Covered
   - Revenue by Region (Map visualization)
   - Top 10 Product-Color Combinations (Bar chart)
   - Monthly Revenue Trends (Line chart)

2. 🌍 REGIONAL ANALYSIS DASHBOARD
   - Regional Performance Comparison (Bar chart)
   - Color Preferences by Region (Heatmap)
   - Skin Tone Alignment Score (Gauge charts)
   - Regional Market Share (Pie chart)

3. 🎨 COLOR PERFORMANCE DASHBOARD
   - Color Performance Matrix (Heatmap)
   - Color Rankings by Product Category (Horizontal bars)
   - Seasonal Color Trends (Line chart with filters)
   - Color Theory vs Actual Performance (Scatter plot)

4. 📦 PRODUCT ANALYSIS DASHBOARD
   - Product Category Performance (Treemap)
   - Best Colors by Product Type (Stacked bars)
   - Price vs Performance Analysis (Scatter plot)
   - Product-Region Performance Matrix

5. 🎯 ALIGNMENT ANALYSIS DASHBOARD (KEY FOR YOUR RESEARCH)
   - Alignment Percentage by Region (Gauge chart)
   - Theoretical vs Actual Color Performance (Side-by-side bars)
   - Optimization Opportunities (Text table)
   - Color Theory Recommendation Heatmap

🔧 KEY TABLEAU FEATURES TO USE:
   - Parameters for dynamic region/product filtering
   - Quick filters for interactivity
   - Calculated fields for alignment scores
   - Tooltips with detailed metrics
   - Actions for dashboard interactivity
"""
print(tableau_specs)

# SECTION 4: DATA QUALITY SUMMARY
print("\n✅ SECTION 4: DATA QUALITY SUMMARY")
print("-" * 50)

print("\n📋 FINAL DATA QUALITY CHECK:")
for name, dataset in datasets_to_export.items():
    missing_values = dataset.isnull().sum().sum()
    duplicate_rows = dataset.duplicated().sum()
    print(f"\n{name}:")
    print(f"   • Rows: {len(dataset):,}")
    print(f"   • Columns: {len(dataset.columns)}")
    print(f"   • Missing values: {missing_values}")
    print(f"   • Duplicate rows: {duplicate_rows}")
    print(f"   • Status: {'✅ Ready' if missing_values == 0 and duplicate_rows == 0 else '⚠️ Needs attention'}")

print(f"\n🎯 TABLEAU CONNECTION READY:")
print(f"   • All datasets exported to '{export_dir}' folder")
print(f"   • Connect Tableau to CSV files for visualization")
print(f"   • Use relationship model to connect datasets")

print("\n" + "="*60)
print("🎉 PROJECT COMPLETION STATUS")
print("="*60)
print("✅ Step 1: Data Loading & Exploration - COMPLETE")
print("✅ Step 2: Data Merging & Metrics Creation - COMPLETE") 
print("✅ Step 3: Exploratory Data Analysis - COMPLETE")
print("✅ Step 4: Core Research Questions Analysis - COMPLETE")
print("✅ Step 5: Tableau Data Preparation - COMPLETE")
print("\n🚀 READY FOR TABLEAU VISUALIZATION!")
print("📊 All datasets prepared and exported")
print("📋 Dashboard specifications provided")
print("🎯 Business insights validated and quantified")