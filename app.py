import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(
    page_title="Vanguard | Executive Sales Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Sidebar Settings & Theme Toggle
with st.sidebar:
    st.markdown("<h3 style='margin-bottom:0;'>❖ Vanguard</h3>", unsafe_allow_html=True)
    st.caption("Sales Intelligence Workspace")
    st.markdown("<hr style='margin: 15px 0;'>", unsafe_allow_html=True)
    
    # Theme Mode Selector
    theme_mode = st.radio("Display Theme", ["Light Mode", "Dark Mode"], horizontal=True)
    
    # Currency Selector
    currency_choice = st.selectbox("Currency Display", ["USD ($)", "INR (₹)"])
    rate = 83.5 if "INR" in currency_choice else 1.0
    curr_symbol = "₹" if "INR" in currency_choice else "$"
    
    st.markdown("---")
    menu_choice = st.radio(
        "Navigation",
       ["📊 Revenue Analytics", "👥 Customer Segmentation", "🎯 Live Deal Scorer", "📁 Data Ledger", "Add New Record"],
        label_visibility="collapsed"
        

    )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Signed in as: **Supreeti (Admin)**")

# 3.# 3. Dynamic Theme Styles (Fixed Light Mode text visibility)
if theme_mode == "Dark Mode":
    bg_color, card_bg, text_color, sub_text, border_col = "#0E1117", "#1E2130", "#FFFFFF", "#94A3B8", "#2E344E"
    chart_bg = "#1E2130"
    plt_palette = ['#818CF8', '#A5B4FC', '#C7D2FE', '#E0E7FF']
else:
    bg_color, card_bg, text_color, sub_text, border_col = "#F8FAFC", "#FFFFFF", "#0F172A", "#475569", "#CBD5E1"
    chart_bg = "#FFFFFF"
    plt_palette = ['#4F46E5', '#6366F1', '#818CF8', '#A5B4FC']



st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Apply dynamic colors via inline injection (with complete light mode text fix)
st.markdown(f"""
<style>
    .stApp {{ background-color: {bg_color}; color: {text_color}; }}
    
    /* Force sidebar background and text visibility in both modes */
    section[data-testid="stSidebar"] {{ 
        background-color: {card_bg} !important; 
        border-right: 1px solid {border_col}; 
        padding-top: 20px; 
    }}
    section[data-testid="stSidebar"] *, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label, 
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stRadio div,
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label {{ 
        color: {text_color} !important; 
    }}

    /* Fix input labels and text inside widgets in Light Mode */
    .stSelectbox div, .stNumberInput label, .stSlider label, .stRadio label {{
        color: {text_color} !important;
    }}

    .dashboard-header {{ background: {card_bg}; border: 1px solid {border_col}; border-radius: 14px; padding: 24px 28px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02); display: flex; justify-content: space-between; align-items: center; }}
    .metric-card {{ background-color: {card_bg}; border: 1px solid {border_col}; border-radius: 12px; padding: 20px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03); }}
    .metric-title {{ font-size: 12px; font-weight: 600; color: {sub_text}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
    .metric-value {{ font-size: 26px; font-weight: 700; color: {text_color}; margin-bottom: 4px; }}
    .metric-badge-pos {{ font-size: 11px; font-weight: 600; color: #10B981; background: #ECFDF5; padding: 2px 8px; border-radius: 6px; display: inline-block; }}
    .content-card {{ background-color: {card_bg}; border: 1px solid {border_col}; border-radius: 14px; padding: 22px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02); }}
    .card-title {{ font-size: 15px; font-weight: 700; color: {text_color}; margin-bottom: 4px; }}
    .card-desc {{ font-size: 12px; color: {sub_text}; margin-bottom: 16px; }}
</style>
""", unsafe_allow_html=True)

# 4. Load Data & Train Model
if 'df' not in st.session_state:
    try:
        st.session_state.df = pd.read_csv('data/sales_data.csv')
    except:
        data = {
            'Customer_Age': [35, 24, 48, 29, 52, 38, 45, 22],
            'Annual_Income': [75000, 42000, 110000, 50000, 130000, 65000, 95000, 35000],
            'Purchase_Amount': [2800, 450, 4200, 600, 4900, 1500, 3100, 300],
            'Product_Category': [0, 1, 3, 1, 3, 2, 0, 1],
            'Previous_Purchases': [5, 2, 12, 1, 15, 4, 9, 0],
            'Account_Tenure': [24, 6, 36, 12, 48, 18, 30, 3],
            'High_Value': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        st.session_state.df = pd.DataFrame(data)

df = st.session_state.df
features = ['Customer_Age', 'Annual_Income', 'Purchase_Amount', 'Product_Category', 'Previous_Purchases', 'Account_Tenure']
X = df[features]
y = df['High_Value']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

importances = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)
accuracy = acc
conf_matrix = cm
category_map = {0: "Electronics", 1: "Apparel", 2: "Home & Living", 3: "Luxury"}

# 5. Main Content Navigation
if menu_choice == "📊 Revenue Analytics":
    st.markdown(f"""
    <div class="dashboard-header">
        <div>
            <h2 style="margin:0; font-size:22px; color:{text_color};">Revenue Analytics & Performance</h2>
            <p style="margin:4px 0 0 0; font-size:13px; color:{sub_text};">Track transactional metrics, revenue flow, and regional distribution.</p>
        </div>
        <div style="background:{border_col}; color:{text_color}; padding:6px 14px; border-radius:8px; font-weight:600; font-size:12px;">
            Currency: {currency_choice}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    total_rev = df['Purchase_Amount'].sum() * rate
    vip_rev = df[df['High_Value'] == 1]['Purchase_Amount'].sum() * rate
    avg_order = df['Purchase_Amount'].mean() * rate
    conversion_rate = (len(df[df['High_Value'] == 1]) / len(df)) * 100

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Revenue</div><div class="metric-value">{curr_symbol}{total_rev:,.0f}</div><span class="metric-badge-pos">+12.4% vs last period</span></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">VIP Segment Revenue</div><div class="metric-value">{curr_symbol}{vip_rev:,.0f}</div><span class="metric-badge-pos">+18.2% high-value growth</span></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average Order Value</div><div class="metric-value">{curr_symbol}{avg_order:,.1f}</div><span class="metric-badge-pos">+5.4% efficiency</span></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">VIP Conversion Rate</div><div class="metric-value">{conversion_rate:.1f}%</div><span class="metric-badge-pos">Random Forest Model</span></div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_l, col_r = st.columns([1.3, 1])
    with col_l:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">Revenue Breakdown by Category ({curr_symbol})</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Total sales volume grouped across retail product divisions.</div>', unsafe_allow_html=True)
        
        df_cat = df.copy()
        df_cat['Category'] = df_cat['Product_Category'].map(category_map)
        cat_rev = df_cat.groupby('Category')['Purchase_Amount'].sum().reset_index()
        cat_rev['Purchase_Amount'] = cat_rev['Purchase_Amount'] * rate
        
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(chart_bg)
        ax.set_facecolor(chart_bg)
        sns.barplot(data=cat_rev, x='Category', y='Purchase_Amount', palette=plt_palette, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(f"Revenue ({curr_symbol})", color=sub_text, fontsize=10)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', alpha=0.4, color=border_col)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_r:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Customer Segment Share</div><div>&nbsp;</div>', unsafe_allow_html=True)
        
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor(chart_bg)
        counts = df['High_Value'].value_counts()
        ax2.pie(
            counts, labels=['Standard Retail', 'High-Value VIP'], autopct='%1.1f%%',
            startangle=90, colors=[border_col, '#4F46E5'],
            wedgeprops=dict(width=0.45, edgecolor=card_bg, linewidth=2),
            textprops=dict(color=text_color, fontsize=10, fontweight='bold')
        )
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu_choice == "👥 Customer Segmentation":
    st.markdown(f'<div class="dashboard-header"><h2 style="margin:0; font-size:22px; color:{text_color};">Customer Segmentation & Cohorts</h2></div>', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">AI Feature Importance Drivers</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Key factors evaluated by the Random Forest classifier.</div>', unsafe_allow_html=True)
        
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        fig3.patch.set_facecolor(chart_bg)
        ax3.set_facecolor(chart_bg)
        importances.plot(kind='barh', color='#4F46E5', ax=ax3)
        ax3.set_xlabel("Weight", color=sub_text)
        ax3.tick_params(colors=text_color, labelsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        st.pyplot(fig3)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_b:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Model Confusion Matrix Evaluation</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-desc">Test set accuracy distribution.</div>', unsafe_allow_html=True)
        
        fig4, ax4 = plt.subplots(figsize=(5.5, 4))
        fig4.patch.set_facecolor(chart_bg)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax4,
                    xticklabels=['Standard', 'VIP'], yticklabels=['Standard', 'VIP'])
        ax4.set_xlabel("Predicted", color=sub_text)
        ax4.set_ylabel("Actual", color=sub_text)
        ax4.tick_params(colors=text_color)
        st.pyplot(fig4)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu_choice == "🔮 Live Deal Scorer":
    st.title("🎯 Live Deal Scorer (External Input)")
    st.write("Type or select customer values below to test the AI model instantly:")
  
# THIS IS THE EXACT POSITION TO PUT YOUR INPUT BOXES:
age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000)
purchase = st.number_input("Purchase Amount ($)", min_value=10, max_value=50000, value=1000)
tenure = st.number_input("Account Tenure (Months)", min_value=1, max_value=120, value=12)

# The button that runs the AI model
if st.button("Run AI Classification"):
    # AI code goes here...
    
    sc1, sc2 = st.columns([1, 1.2])
    with sc1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Input Customer Parameters</div>', unsafe_allow_html=True)
        
        age = st.slider("Customer Age", 18, 70, 35)
        income = st.number_input(f"Annual Income ({curr_symbol})", int(20000*rate), int(150000*rate), int(75000*rate), step=int(5000*rate)) / rate
        purchase_amt = st.number_input(f"Transaction Value ({curr_symbol})", int(100*rate), int(5000*rate), int(2500*rate), step=int(100*rate)) / rate
        category = st.selectbox("Product Category", options=[0, 1, 2, 3], format_func=lambda x: category_map[x])
        prev_purchases = st.slider("Historical Orders", 0, 20, 5)
        
        score_btn = st.button("Run AI Classification", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with sc2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Classification Output & Recommendation</div>', unsafe_allow_html=True)
        
        if score_btn:
            input_data = np.array([[age, income, purchase_amt, category, prev_purchases]])
            prediction = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1] * 100
            
            if prediction == 1:
                st.success(f"### ⭐ High-Value VIP Account\n**Confidence Score:** `{prob:.1f}%`\n\n*Recommendation:* Route directly to priority concierge support and unlock loyalty tier incentives.")
            else:
                st.info(f"### 📈 Standard Retail Tier\n**VIP Probability:** `{prob:.1f}%`\n\n*Recommendation:* Enroll customer in automated up-sell and product recommendation email series.")
        else:
            st.info("Adjust the parameters on the left and click **Run AI Classification** to analyze a transaction.")
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown(f'<div class="dashboard-header"><h2 style="margin:0; font-size:22px; color:{text_color};">Transaction Ledger & CSV Aggregator</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    
    disp_df = df.copy()
    disp_df['Purchase_Amount'] = disp_df['Purchase_Amount'] * rate
    disp_df['Category'] = disp_df['Product_Category'].map(category_map)
    disp_df['Account Tier'] = disp_df['High_Value'].map({1: 'High-Value VIP', 0: 'Standard'})
    
    st.markdown(f"**Showing transaction amounts converted to {currency_choice}:**")
    st.dataframe(disp_df[['Customer_Age', 'Annual_Income', 'Purchase_Amount', 'Category', 'Previous_Purchases', 'Account Tier']].head(50), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)




if "Add New Record" in menu_choice:
    st.title("➕ Add New Transaction to Dataset")
    
    with st.form("new_data_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_age = st.number_input("Customer Age", 18, 90, 30)
            new_income = st.number_input(f"Annual Income ({curr_symbol})", 10000, 500000, int(60000 * rate)) / rate
            new_purchase = st.number_input(f"Purchase Amount ({curr_symbol})", 10, 50000, int(1500 * rate)) / rate
        with col2:
            new_cat = st.selectbox("Product Category", [0, 1, 2, 3])
            new_prev = st.number_input("Previous Purchases", 0, 50, 3)
            new_tenure = st.number_input("Account Tenure (Months)", 1, 120, 12)
            new_label = st.selectbox("Is High-Value VIP?", [1, 0])
            
        submit_button = st.form_submit_button("Add to Dataset & Retrain AI")
        
        if submit_button:
            new_row = pd.DataFrame([[new_age, new_income, new_purchase, new_cat, new_prev, new_tenure, new_label]], 
                                    columns=['Customer_Age', 'Annual_Income', 'Purchase_Amount', 'Product_Category', 'Previous_Purchases', 'Account_Tenure', 'High_Value'])
            
            st.session_state.df = pd.concat([st.session_state.df, new_row], ignore_index=True)
            st.session_state.df.to_csv("data/sales_data.csv", index=False)
            
            st.success("🎉 Success! The new record has been added to the CSV file, and the AI model has automatically retrained!")