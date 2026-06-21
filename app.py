import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


kmeans = joblib.load('kmeans.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load("label_encoders.pkl")
df_encoded = pd.read_csv(r"E:\Project4\clustered_mobile_data.csv")
df=pd.read_csv(r"cleaned_mobile_data.csv")
scaler_rec = joblib.load("recommendation_scaler.pkl")
X = joblib.load("X.pkl")
product_df = pd.read_csv("product_df.csv")
# Sidebar Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Recommendation"]
)
# Recommendation system Function Block Based on Cosine Similarity
def recommend_product(brand, price_usd, rating, top_n=10):

    # encode brand safely
    if isinstance(brand, str):
        brand = label_encoders['brand'].transform([brand])[0]

    # scale input
    user_vec = scaler_rec.transform([[brand, price_usd, rating]])

    # cosine similarity
    user_sim = cosine_similarity(user_vec, X)

    sim_scores = list(enumerate(user_sim[0]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    indices = [i[0] for i in sim_scores]

    result = product_df.iloc[indices][['brand','model','rating','price_usd']].copy()

    result['Similarity'] = [i[1] for i in sim_scores]

    result['brand'] = label_encoders['brand'].inverse_transform(result['brand'].astype(int))
    result['model'] = label_encoders['model'].inverse_transform(result['model'].astype(int))

    return result
def section_header(icon, title, color="#4F86C6"):
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, {color} 0%, {color}99 100%);
        padding: 10px 20px; border-radius: 8px; margin: 15px 0 10px 0;">
        <h3 style="color: white; margin: 0; font-size: 17px; font-weight: 600;">
            {icon} {title}
        </h3>
    </div>
    """, unsafe_allow_html=True)
if page == "Dashboard":
    # DASHBOARD CODE
    
    # SIDEBAR FILTERS
    
    st.sidebar.header("🔍 Filters")

    # Age Filter
    age_range = st.sidebar.slider(
        "Select Age Range",
        int(df_encoded['age'].min()),
        int(df_encoded['age'].max()),
        (
            int(df_encoded['age'].min()),
            int(df_encoded['age'].max())
        )
    )

    # BRAND FILTER
    
    brand_names = ["All"] + list(
        label_encoders['brand'].inverse_transform(
            sorted(df_encoded['brand'].unique())
        )
    )

    selected_brand_name = st.sidebar.selectbox(
        "Select Brand",
        brand_names
    )

    # MODEL FILTER

    if selected_brand_name == "All":

        model_names = ["All"] + list(
            label_encoders['model'].classes_
        )

    else:

        brand_encoded = label_encoders['brand'].transform(
            [selected_brand_name]
        )[0]

        model_encoded_list = sorted(
            df_encoded[
                df_encoded['brand'] == brand_encoded
            ]['model'].unique()
        )

        model_names = ["All"] + list(
            label_encoders['model'].inverse_transform(
                model_encoded_list
            )
        )

    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        model_names
    )

    # CLUSTER FILTER

    cluster_options = ["All"] + list(
        sorted(df_encoded['Cluster'].unique())
    )

    selected_cluster = st.sidebar.selectbox(
        "Select Cluster",
        cluster_options
    )

    # COUNTRY FILTER
    
    country_names = ["All"] + list(
        label_encoders['country'].inverse_transform(
            sorted(df_encoded['country'].unique())
        )
    )

    selected_country_name = st.sidebar.selectbox(
        "Select Country",
        country_names
    )  
    # APPLY FILTERS
    
    filtered_df = df_encoded.copy()

    # Age Filter
    filtered_df = filtered_df[
        filtered_df['age'].between(
            age_range[0],
            age_range[1]
        )
    ]

    # Brand Filter
    if selected_brand_name != "All":

        brand_encoded = label_encoders['brand'].transform(
            [selected_brand_name]
        )[0]

        filtered_df = filtered_df[
            filtered_df['brand'] == brand_encoded
        ]

    # Model Filter
    if selected_model_name != "All":

        model_encoded = label_encoders['model'].transform(
            [selected_model_name]
        )[0]

        filtered_df = filtered_df[
            filtered_df['model'] == model_encoded
        ]

    # Cluster Filter
    if selected_cluster != "All":

        filtered_df = filtered_df[
            filtered_df['Cluster'] == selected_cluster
        ]
        
    # Country Filter
    if selected_country_name != "All":

        country_encoded = label_encoders['country'].transform(
            [selected_country_name]
        )[0]

        filtered_df = filtered_df[
            filtered_df['country'] == country_encoded
        ]

    plt.style.use('default')

    # ── CUSTOM COLORS ─────────────────────────────────────────
    COLORS     = ['#4F86C6', '#F4845F', '#58B368']   # blue, orange, green — consistent across all charts
    GRID_STYLE = dict(axis='y', linestyle='--', alpha=0.4, color='#cccccc')
    FIG_SIZE   = (4.6, 3.8)
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #2563EB 50%, #4F86C6 100%);
        padding: 24px 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
        display: flex;
        align-items: center;
        gap: 16px;
    ">
        <div style="
            background: rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 12px;
            font-size: 36px;
            line-height: 1;
        ">📱</div>
        <div>
            <div style="
                color: rgba(255,255,255,0.75);
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin-bottom: 4px;
            ">E-Commerce Analytics</div>
            <div style="
                color: white;
                font-size: 26px;
                font-weight: 800;
                letter-spacing: 0.5px;
                line-height: 1.2;
            ">Mobile Customer Intelligence Dashboard</div>
        </div>
    </div>
""", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)

    card_style = """
        <div style="
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            text-align: center;
            border-top: 4px solid #2563EB;
            border-left: 1px solid #e5e7eb;
            border-right: 1px solid #e5e7eb;
            border-bottom: 1px solid #e5e7eb;
        ">
            <div style="font-size:14px; color:gray;">{label}</div>
            <div style="font-size:22px; font-weight:bold; color:#111827; margin-top:5px;">
                {value}
            </div>
        </div>
    """

    with k1:
        st.markdown(card_style.format(label="👥 Total Users", value=f"{len(filtered_df):,}"), unsafe_allow_html=True)

    with k2:
        st.markdown(card_style.format(label="💵 Avg Price", value=f"${filtered_df['price_usd'].mean():.2f}"), unsafe_allow_html=True)

    with k3:
        st.markdown(card_style.format(label="⭐ Avg Rating", value=f"{filtered_df['rating'].mean():.2f}"), unsafe_allow_html=True)

    with k4:
        st.markdown(card_style.format(label="🔗 Clusters", value=filtered_df['Cluster'].nunique()), unsafe_allow_html=True)

    # ── HELPER : add value labels on bars ─────────────────────
    def add_bar_labels(ax, fmt="{:.0f}", color="#333333", fontsize=9):
        for bar in ax.patches:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + ax.get_ylim()[1] * 0.01,
                    fmt.format(h),
                    ha='center', va='bottom',
                    fontsize=fontsize, fontweight='bold', color=color
                )

    # ══════════════════════════════════════════════════════════
    # ROW 1 : Cluster Pie | Sentiment Donut | Sentiment by Cluster
    # ══════════════════════════════════════════════════════════
    section_header("📊", "Customer & Sentiment Overview", "#4F86C6")
    
    col1, col2, col3 = st.columns(3)

    temp_df = filtered_df.copy()
    temp_df['Sentiment_Label'] = temp_df['sentiment'].map(
        {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    )

    with col1:
        cluster_counts = filtered_df['Cluster'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        wedges, texts, autotexts = ax.pie(
            cluster_counts,
            labels=[f'Cluster {i}' for i in cluster_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            colors=COLORS[:len(cluster_counts)],
            textprops={'fontsize': 9},
            wedgeprops={'edgecolor': 'white', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_fontweight('bold')
        ax.set_title("Cluster Distribution", fontsize=11, fontweight='bold', pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        sentiment_counts = temp_df['Sentiment_Label'].value_counts()
        total = sentiment_counts.sum()
        #SENT_COLORS = ['#E74C3C', '#F39C12', '#27AE60']
        SENT_COLORS = ['#27AE60', '#F39C12', '#E74C3C']
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=SENT_COLORS[:len(sentiment_counts)],
            textprops={'fontsize': 9},
            wedgeprops={'width': 0.5, 'edgecolor': 'white', 'linewidth': 2}
        )
        ax.text(0, 0, f"{total:,}\nUsers", ha='center', va='center',
                fontsize=11, fontweight='bold', color='#333')
        ax.set_title("Sentiment Distribution", fontsize=11, fontweight='bold', pad=10)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col3:
        sentiment_percent = pd.crosstab(
            filtered_df['Cluster'], temp_df['Sentiment_Label'], normalize='index'
        ) * 100
        fig, ax = plt.subplots(figsize=(4.6, 4.4))
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        sentiment_percent.plot(
            kind='bar', stacked=True, ax=ax, width=0.6,
            color=['#E74C3C', '#F39C12', '#27AE60'],
            edgecolor='white'
        )
        ax.set_title("Sentiment by Cluster", fontsize=11, fontweight='bold')
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.legend(title="Sentiment", loc="upper right", fontsize=7, title_fontsize=8)
        ax.grid(**GRID_STYLE)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        
    # ══════════════════════════════════════════════════════════
    # ROW 2 : Cluster Countplot | Avg Price | Avg Rating
    # ══════════════════════════════════════════════════════════
    section_header("🧩", "Cluster Analysis", "#4F86C6")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')
        cluster_counts2 = filtered_df['Cluster'].value_counts().sort_index()
        bars = ax.bar(
            [f'C{i}' for i in cluster_counts2.index],
            cluster_counts2.values,
            color=COLORS[:len(cluster_counts2)],
            width=0.5, edgecolor='white', linewidth=1.5
        )
        add_bar_labels(ax, fmt="{:.0f}")
        ax.set_title("Customer Distribution", fontsize=11, fontweight='bold')
        ax.set_xlabel("Cluster", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(**GRID_STYLE)
        ax.spines[['top','right']].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        cluster_price = filtered_df.groupby('Cluster')['price_usd'].mean()
        if cluster_price.empty:
            st.warning("No data available")
        else:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            bars = ax.bar(
                [f'C{i}' for i in cluster_price.index],
                cluster_price.values,
                color=COLORS[:len(cluster_price)],
                width=0.5, edgecolor='white', linewidth=1.5
            )
            add_bar_labels(ax, fmt="${:.0f}")
            ax.set_title("Avg Price by Cluster", fontsize=11, fontweight='bold')
            ax.set_xlabel("Cluster", fontsize=9)
            ax.set_ylabel("Avg Price (USD)", fontsize=9)
            ax.grid(**GRID_STYLE)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with col3:
        cluster_rating = filtered_df.groupby('Cluster')['rating'].mean()
        if cluster_rating.empty:
            st.warning("No data available")
        else:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            bars = ax.bar(
                [f'C{i}' for i in cluster_rating.index],
                cluster_rating.values,
                color=COLORS[:len(cluster_rating)],
                width=0.5, edgecolor='white', linewidth=1.5
            )
            add_bar_labels(ax, fmt="{:.2f}")
            ax.set_title("Avg Rating by Cluster", fontsize=11, fontweight='bold')
            ax.set_xlabel("Cluster", fontsize=9)
            ax.set_ylabel("Avg Rating", fontsize=9)
            ax.set_ylim(0, 5.5)
            ax.grid(**GRID_STYLE)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ══════════════════════════════════════════════════════════
    # ROW 3 : Brand Distribution | Country Distribution | Heatmap
    # ══════════════════════════════════════════════════════════
    section_header("🌍", "Brand, Country & Feature Overview", "#4F86C6")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        brand_cluster = pd.crosstab(filtered_df['Cluster'], filtered_df['brand'])
        if brand_cluster.empty:
            st.warning("No data available")
        else:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            brand_cluster.plot(kind='bar', ax=ax, width=0.6, edgecolor='white')
            ax.set_title("Brand Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Cluster", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(fontsize=6, loc='upper right', title='Brand', title_fontsize=7)
            ax.grid(**GRID_STYLE)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with col2:
        country_cluster = pd.crosstab(filtered_df['Cluster'], filtered_df['country'])
        if country_cluster.empty:
            st.warning("No data available")
        else:
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            fig.patch.set_facecolor('#F8F9FA')
            ax.set_facecolor('#F8F9FA')
            country_cluster.plot(kind='bar', ax=ax, width=0.6, edgecolor='white')
            ax.set_title("Country Distribution", fontsize=11, fontweight='bold')
            ax.set_xlabel("Cluster", fontsize=9)
            ax.set_ylabel("Count", fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.legend(fontsize=6, loc='upper right', title='Country', title_fontsize=7)
            ax.grid(**GRID_STYLE)
            ax.spines[['top','right']].set_visible(False)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with col3:
        cluster_summary = (
            filtered_df.groupby('Cluster')[
                ['price_usd','rating','battery_life_rating','camera_rating',
                'performance_rating','design_rating','display_rating']
            ].mean().round(2)
        )
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        fig.patch.set_facecolor('#F8F9FA')
        sns.heatmap(
            cluster_summary, annot=True, cmap='Blues',
            fmt='.2f', ax=ax, annot_kws={'size': 7},
            linewidths=0.5, linecolor='white',
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title("Feature Heatmap", fontsize=11, fontweight='bold')
        ax.tick_params(axis='x', labelsize=7, rotation=30)
        ax.tick_params(axis='y', labelsize=8, rotation=0)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    
    
    # ══════════════════════════════════════════════════════════
    # ROW 4 : Price vs Rating | Feature Ratings | Verified vs Sentiment
    # ══════════════════════════════════════════════════════════
    section_header("🎯", "Price, Features & Purchase Behavior", "#4F86C6")
    col1, col2, col3 = st.columns(3)

    # ─────────────────────────────
    # 1. PRICE vs RATING
    # ─────────────────────────────
    with col1:
        fig, ax = plt.subplots(figsize=FIG_SIZE)          
        fig.patch.set_facecolor('#F8F9FA')                
        ax.set_facecolor('#F8F9FA')                       

        sns.scatterplot(
            data=filtered_df,
            x='price_usd',
            y='rating',
            hue='Cluster',
            palette=COLORS[:filtered_df['Cluster'].nunique()],  
            ax=ax,
            legend=False
        )

        ax.set_title("Price vs Rating", fontsize=11, fontweight='bold')
        ax.set_xlabel("Price (USD)", fontsize=9)
        ax.set_ylabel("Rating", fontsize=9)
        ax.spines[['top','right']].set_visible(False)    
        ax.grid(**GRID_STYLE)                             

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


    # ─────────────────────────────
    # 2. FEATURE PERFORMANCE
    # ─────────────────────────────
    with col2:
        feature_cols = [
            'battery_life_rating',
            'camera_rating',
            'performance_rating',
            'design_rating',
            'display_rating'
        ]

        feature_avg = filtered_df[feature_cols].mean()

        fig, ax = plt.subplots(figsize=FIG_SIZE)          
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')

        feature_avg.plot(
            kind='bar',
            ax=ax,
            color=COLORS[0],                              
            width=0.5,
            edgecolor='white'
        )

        add_bar_labels(ax, fmt="{:.2f}")                  

        ax.set_title("Feature Ratings", fontsize=11, fontweight='bold')
        ax.set_ylabel("Avg Rating", fontsize=9)
        ax.tick_params(axis='x', rotation=30, labelsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(**GRID_STYLE)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


    # ─────────────────────────────
    # 3. VERIFIED vs SENTIMENT
    # ─────────────────────────────
    with col3:
        temp = filtered_df.copy()

        pivot = pd.crosstab(
            temp['verified_purchase'],
            temp['sentiment'],
            normalize='index'
        ) * 100

        fig, ax = plt.subplots(figsize=FIG_SIZE)          
        fig.patch.set_facecolor('#F8F9FA')
        ax.set_facecolor('#F8F9FA')

        pivot.plot(
            kind='bar',
            stacked=True,
            ax=ax,
            color=['#E74C3C', '#F39C12', '#27AE60'],      
            edgecolor='white',
            width=0.5
        )

        ax.set_title("Verified vs Sentiment", fontsize=11, fontweight='bold')
        ax.set_xlabel("Verified Purchase (0 = FALSE, 1 = TRUE)", fontsize=9)
        ax.set_ylabel("Percentage (%)", fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)  

        ax.legend(
            ['Negative', 'Neutral', 'Positive'],
            title="Sentiment",
            fontsize=7,
            title_fontsize=8,
            loc="upper right"
        )

        ax.spines[['top','right']].set_visible(False)
        ax.grid(**GRID_STYLE)

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    # ══════════════════════════════════════════════════════════
    # CLUSTER PROFILE TABLE
    # ══════════════════════════════════════════════════════════
    section_header("📋", "Cluster Profile Summary", "#4F86C6")
    st.dataframe(
        cluster_summary.style
        .background_gradient(cmap='Blues', axis=None)
        .format("{:.2f}"),
        use_container_width=True
    )

elif page == "Recommendation":
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.1);
    ">
        <div style="color:rgba(255,255,255,0.5); font-size:11px; letter-spacing:3px; text-transform:uppercase; margin-bottom:8px;">
            🛒 Product Discovery
        </div>
        <div style="color:white; font-size:28px; font-weight:800;">
            📱 Mobile Recommendation System
        </div>
    </div>
""", unsafe_allow_html=True)

    brand_name = st.selectbox(
        "Select Brand",
        label_encoders['brand'].classes_
    )

    price_input = st.number_input(
        "Enter Price",
        min_value=0.0,
        value=700.0
    )

    rating_input = st.selectbox(
        "Rating",
        [1, 2, 3, 4, 5]
    )

    if st.button("Recommend"):

        result = recommend_product(
            brand_name,
            price_input,
            rating_input
        )

        st.success("Top Recommendations")

        st.dataframe(result, use_container_width=True)
    