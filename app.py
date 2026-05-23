import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, MultiLabelBinarizer, LabelEncoder

# ==========================================
# 1. PAGE SETUP & WIDESCREEN INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="Tying the Data Knot: Love, Life & Likes",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS variables to support dynamic Dark/Light modes
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,600;1,600&display=swap');
    
    /* Core Typography & Base Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .gradient-header {
        background: linear-gradient(135deg, #ff2a5f 0%, #7e57c2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 5px;
    }
    
    .serif-subtitle {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 1.25rem;
        color: var(--text-color);
        opacity: 0.85;
        margin-bottom: 25px;
    }
    
    /* Glassmorphic Container Cards */
    .premium-card {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    
    .premium-card:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    .card-title {
        font-weight: 700;
        font-size: 1.1rem;
        color: #ff2a5f;
        margin-bottom: 15px;
        border-bottom: 2px solid rgba(255, 42, 95, 0.1);
        padding-bottom: 5px;
    }
    
    /* Custom simulation result display card */
    .prediction-card {
        border-radius: 16px;
        padding: 30px;
        color: white;
        text-align: center;
        margin-top: 15px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
        transition: all 0.3s ease;
    }
    
    .prob-bar-container {
        background-color: rgba(255, 255, 255, 0.15);
        border-radius: 10px;
        height: 10px;
        width: 100%;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .prob-bar-fill {
        background-color: white;
        height: 100%;
        border-radius: 10px;
    }

    /* Structured Alert Callout */
    .critique-box {
        background-color: rgba(126, 87, 194, 0.05);
        border-left: 5px solid #7e57c2;
        border-radius: 4px 16px 16px 4px;
        padding: 20px;
        margin: 20px 0;
    }
    
    /* Interactive micro-animations */
    button.step-btn {
        transition: all 0.2s ease-in-out;
    }
    button.step-btn:hover {
        transform: scale(1.03);
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. CACHING ASSETS & DATA LOADERS
# ==========================================
@st.cache_data
def load_dataset():
    """Loads and caches the raw CSV dataset."""
    df = pd.read_csv('dating_app_behavior_dataset_extended.csv')
    return df

@st.cache_resource
def load_ml_model():
    """Loads and caches the LightGBM classifier pickle."""
    with open('final_model.pkl', 'rb') as f:
        return joblib.load(f)

@st.cache_resource
def get_preprocessing_pipeline(_df):
    """
    Fits and caches the scikit-learn preprocessing pipeline.
    Recreates the exact training preprocessing from the notebooks.
    """
    # 1. Clean dataset copies
    df_clean = _df.copy()
    df_clean.drop_duplicates(inplace=True)
    df_clean = df_clean.drop(columns=['app_usage_time_label', 'swipe_right_label'], errors='ignore')
    
    # 2. MultiLabelBinarizer for 'interest_tags'
    mlb = MultiLabelBinarizer()
    df_clean['interest_tags'] = df_clean['interest_tags'].fillna('')
    interests_list = df_clean['interest_tags'].apply(lambda x: [tag.strip() for tag in x.split(',')])
    mlb.fit(interests_list)
    df_interests = pd.DataFrame(mlb.transform(interests_list), columns=[f"interest_{c}" for c in mlb.classes_], index=df_clean.index)
    df_processed = pd.concat([df_clean, df_interests], axis=1).drop(columns=['interest_tags'])
    
    # 3. Cyclic Transformation
    df_processed['last_active_hour_sin'] = np.sin(2 * np.pi * df_processed['last_active_hour'] / 24)
    df_processed['last_active_hour_cos'] = np.cos(2 * np.pi * df_processed['last_active_hour'] / 24)
    df_processed = df_processed.drop(columns=['last_active_hour'])
    
    # 4. Outlier Capping Limits
    cols_to_cap = ['app_usage_time_min', 'message_sent_count', 'likes_received', 'mutual_matches']
    cap_limits = {}
    for col in cols_to_cap:
        cap_limits[col] = (df_processed[col].quantile(0.01), df_processed[col].quantile(0.99))
        df_processed[col] = df_processed[col].clip(lower=cap_limits[col][0], upper=cap_limits[col][1])
        
    X = df_processed.drop(columns=['match_outcome'])
    y = df_processed['match_outcome']
    
    # Stratified Split (matching random state 42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # ColumnTransformer Setup
    ordinal_cols = ['education_level', 'income_bracket']
    nominal_cols = ['gender', 'sexual_orientation', 'location_type', 'body_type', 'relationship_intent', 'swipe_time_of_day', 'zodiac_sign']
    numeric_cols = df_processed.select_dtypes(include=['number']).columns.drop('match_outcome', errors='ignore').tolist()
    numeric_cols = [col for col in numeric_cols if not col.startswith('interest_')]
    
    education_order = ['No Formal Education', 'High School', 'Diploma', 'Associate’s', 'Bachelor’s', 'Master’s', 'MBA', 'PhD', 'Postdoc']
    income_order = ['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(categories=[education_order, income_order]), ordinal_cols),
            ('nominal', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), nominal_cols),
            ('scaler', RobustScaler(), numeric_cols)
        ],
        remainder='passthrough'
    )
    
    # Fit on training split features
    preprocessor.fit(X_train)
    
    # Label encoder for target class mapping
    le = LabelEncoder()
    le.fit(y_train)
    
    return preprocessor, mlb, cap_limits, le, list(X_train.columns)


# Load assets
try:
    df_raw = load_dataset()
    model = load_ml_model()
    preprocessor, mlb, cap_limits, le, X_train_cols = get_preprocessing_pipeline(df_raw)
except Exception as e:
    st.error(f"Error loading system assets or datasets: {e}")
    st.stop()


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.markdown(
    '<div style="text-align: center;"><span style="font-size: 3rem;">💘</span></div>', 
    unsafe_allow_html=True
)
st.sidebar.markdown(
    '<div style="text-align: center; font-weight:800; font-size:1.35rem; color:#ff2a5f; margin-bottom: 20px;">Tying the Data Knot</div>', 
    unsafe_allow_html=True
)

navigation = st.sidebar.radio(
    "Select App View",
    options=[
        "🔮 Scenario & Match Simulator",
        "📊 Behavioral Contrast Engine",
        "📉 Model Benchmarks & Metrics"
    ]
)




# ==========================================
# VIEW 1: 🔮 SCENARIO & MATCH SIMULATOR
# ==========================================
if navigation == "🔮 Scenario & Match Simulator":
    st.markdown('<div class="gradient-header">Scenario & Match Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-subtitle">Test digital interactions and predict the destiny of simulated couples</div>', unsafe_allow_html=True)
    
    with st.expander("⚙️ Under the Hood: Mathematical Preprocessing Pipeline", expanded=False):
        st.markdown(r"""
        To prevent feature distribution drift and ensure zero prediction skew, the interactive simulator's input pipeline is mathematically aligned with the training configurations:
        
        1. **Outlier Capping:** App telemetry inputs (`app_usage_time_min`, `message_sent_count`, `likes_received`, and `mutual_matches`) are clipped using the exact 1st and 99th percentile bounds calculated from the training split.
        2. **Cyclic Time Conversions:** The user's active hour input ($H \in [0, 23]$) is mapped to circular dimensions using $\sin(2\pi H / 24)$ and $\cos(2\pi H / 24)$ trigonometric transforms.
        3. **Ordinal Mapping:** Education levels and income brackets are encoded using a pre-fit `OrdinalEncoder` using the exact hierarchical categories defined during training.
        4. **One-Hot Nominal Encoding:** Context variables (gender, orientation, zodiac, location, body type, intent) are expanded into dummy indicator variables via `OneHotEncoder`.
        5. **Robust Scaling:** Numeric metrics are scaled via `RobustScaler` using the medians and Interquartile Ranges (IQR) of the training dataset split, rather than standard normalizing.
        6. **Mutual Information Subset:** Out of the resulting 114 preprocessed features, the app extracts the exact 30 features selected by Mutual Information to match the LightGBM input vector.
        """)
    
    # Organize columns dividing Demographics from App Habits
    col_dem, col_hab = st.columns([1, 1], gap="large")
    
    with col_dem:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">👤 User Demographics & Background</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", options=sorted(df_raw['gender'].unique()))
            age = st.slider("Age", min_value=18, max_value=59, value=25)
            education_level = st.selectbox("Education Level", options=['No Formal Education', 'High School', 'Diploma', 'Associate’s', 'Bachelor’s', 'Master’s', 'MBA', 'PhD', 'Postdoc'], index=4)
            income_bracket = st.selectbox("Income Bracket", options=['Very Low', 'Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High', 'Very High'], index=3)
        with c2:
            sexual_orientation = st.selectbox("Sexual Orientation", options=sorted(df_raw['sexual_orientation'].unique()))
            height_cm = st.slider("Height (cm)", min_value=145, max_value=200, value=170)
            weight_kg = st.slider("Weight (kg)", min_value=37.0, max_value=120.0, value=65.0, step=0.5)
            zodiac_sign = st.selectbox("Zodiac Sign", options=sorted(df_raw['zodiac_sign'].unique()))
            
        c3, c4 = st.columns(2)
        with c3:
            body_type = st.selectbox("Body Type", options=sorted(df_raw['body_type'].unique()))
        with c4:
            relationship_intent = st.selectbox("Relationship Intent", options=sorted(df_raw['relationship_intent'].unique()))
            
        # Multi-select interests
        unique_interests = sorted(list(mlb.classes_))
        interest_tags = st.multiselect(
            "Interest Tags",
            options=unique_interests,
            default=['Coding', 'Traveling', 'Music', 'Fitness']
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_hab:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📱 Dating App Habits & Telemetry</div>', unsafe_allow_html=True)
        
        c5, c6 = st.columns(2)
        with c5:
            app_usage_time_min = st.slider("App Usage Time (min/day)", min_value=0, max_value=300, value=120)
            swipe_right_ratio = st.slider("Swipe Right Ratio", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
            likes_received = st.slider("Likes Received", min_value=0, max_value=200, value=50)
            profile_pics_count = st.slider("Profile Pics Count", min_value=0, max_value=6, value=4)
        with c6:
            mutual_matches = st.slider("Mutual Matches", min_value=0, max_value=30, value=5)
            bio_length = st.slider("Bio Length (characters)", min_value=0, max_value=500, value=150)
            message_sent_count = st.slider("Message Sent Count", min_value=0, max_value=100, value=30)
            emoji_usage_rate = st.slider("Emoji Usage Rate", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
            
        c7, c8 = st.columns(2)
        with c7:
            last_active_hour = st.slider("Last Active Hour (0-23)", min_value=0, max_value=23, value=20)
        with c8:
            swipe_time_of_day = st.selectbox("Swipe Time of Day", options=sorted(df_raw['swipe_time_of_day'].unique()))
            
        location_type = st.selectbox("Location Type", options=sorted(df_raw['location_type'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
        
    # Central Submission Button
    st.markdown("<div style='text-align: center; margin-top: 10px; margin-bottom: 25px;'>", unsafe_allow_html=True)
    run_sim = st.button("🔮 Run Match Outcome Simulation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if run_sim:
        with st.spinner("Processing scenario parameters and querying LightGBM pipeline..."):
            try:
                # 1. Assemble single-row raw DataFrame mimicking original CSV structure
                raw_dict = {
                    'gender': [gender],
                    'sexual_orientation': [sexual_orientation],
                    'location_type': [location_type],
                    'income_bracket': [income_bracket],
                    'education_level': [education_level],
                    'interest_tags': [', '.join(interest_tags)],
                    'app_usage_time_min': [app_usage_time_min],
                    'swipe_right_ratio': [swipe_right_ratio],
                    'likes_received': [likes_received],
                    'mutual_matches': [mutual_matches],
                    'profile_pics_count': [profile_pics_count],
                    'bio_length': [bio_length],
                    'message_sent_count': [message_sent_count],
                    'emoji_usage_rate': [emoji_usage_rate],
                    'last_active_hour': [last_active_hour],
                    'swipe_time_of_day': [swipe_time_of_day],
                    'age': [age],
                    'height_cm': [height_cm],
                    'weight_kg': [weight_kg],
                    'zodiac_sign': [zodiac_sign],
                    'body_type': [body_type],
                    'relationship_intent': [relationship_intent]
                }
                input_df = pd.DataFrame(raw_dict)
                
                # 2. Re-create training feature engineering
                # Transform interest_tags into binary columns via fitted MLB
                transformed_interests = mlb.transform([interest_tags])
                df_interests = pd.DataFrame(
                    transformed_interests, 
                    columns=[f"interest_{c}" for c in mlb.classes_], 
                    index=input_df.index
                )
                input_processed = pd.concat([input_df, df_interests], axis=1).drop(columns=['interest_tags'])
                
                # Cyclic active hour sin/cos
                input_processed['last_active_hour_sin'] = np.sin(2 * np.pi * input_processed['last_active_hour'] / 24)
                input_processed['last_active_hour_cos'] = np.cos(2 * np.pi * input_processed['last_active_hour'] / 24)
                input_processed = input_processed.drop(columns=['last_active_hour'])
                
                # Clip outliers using fitted training bounds
                for col, (lower, upper) in cap_limits.items():
                    input_processed[col] = input_processed[col].clip(lower=lower, upper=upper)
                    
                # Reorder features to match exact shape passed to preprocessor
                input_processed = input_processed[X_train_cols]
                
                # 3. Transform using Pipeline's ColumnTransformer
                processed_arr = preprocessor.transform(input_processed)
                
                # Build intermediate DataFrame
                new_nominal_cols = preprocessor.named_transformers_['nominal'].get_feature_names_out(
                    ['gender', 'sexual_orientation', 'location_type', 'body_type', 'relationship_intent', 'swipe_time_of_day', 'zodiac_sign']
                )
                ordinal_cols = ['education_level', 'income_bracket']
                numeric_cols = [
                    'app_usage_time_min', 'swipe_right_ratio', 'likes_received', 'mutual_matches',
                    'profile_pics_count', 'bio_length', 'message_sent_count', 'emoji_usage_rate',
                    'age', 'height_cm', 'weight_kg', 'last_active_hour_sin', 'last_active_hour_cos'
                ]
                interest_cols = [col for col in X_train_cols if col.startswith('interest_')]
                
                all_new_cols = ordinal_cols + list(new_nominal_cols) + numeric_cols + interest_cols
                
                final_df = pd.DataFrame(processed_arr, columns=all_new_cols)
                
                # LightGBM sanitized spaces in feature names -> replace spaces with underscores
                final_df.columns = [c.replace(' ', '_') for c in final_df.columns]
                
                # Select the exact 30 features
                model_features = model.feature_name_
                final_mi_df = final_df[model_features]
                
                # 4. Model Predictions & Probability mapping
                pred_idx = model.predict(final_mi_df)[0]
                pred_label = le.inverse_transform([pred_idx])[0]
                pred_probs = model.predict_proba(final_mi_df)[0]
                
                # Display outcome card beautifully
                # Color code styling: Green for positive, Yellow/Orange for neutral/cautious, Red for warning
                outcomes_meta = {
                    'Blocked': {'color': '#f44336', 'icon': '🚫', 'desc': 'The connection ended abruptly. You have been blocked by the user.'},
                    'Catfished': {'color': '#e65100', 'icon': '🎣', 'desc': 'Beware! The user’s profile details and interaction telemetry represent a synthetic profile.'},
                    'Chat Ignored': {'color': '#757575', 'icon': '👻', 'desc': 'You sent a message, but it remains unread and unanswered.'},
                    'Date Happened': {'color': '#4caf50', 'icon': '🍕', 'desc': 'Success! The digital chemistry translated into a real-world meeting.'},
                    'Ghosted': {'color': '#e0e0e0', 'icon': '🌫️', 'desc': 'The conversation faded out. The user stopped responding without warning.'},
                    'Instant Match': {'color': '#ffeb3b', 'color_text': '#000000', 'icon': '⚡', 'desc': 'Immediate connection! You both swiped right and matched instantly.'},
                    'Mutual Match': {'color': '#e91e63', 'icon': '💖', 'desc': 'It’s a Match! Both of you swiped right on each other’s profiles.'},
                    'No Action': {'color': '#9e9e9e', 'icon': '💤', 'desc': 'Nothing happened. The profile was skipped or remained inactive.'},
                    'One-sided Like': {'color': '#2196f3', 'icon': '💘', 'desc': 'You swiped right, but they did not swipe back.'},
                    'Relationship Formed': {'color': '#9c27b0', 'icon': '💍', 'desc': 'Love wins! The match progressed to a meaningful relationship.'}
                }
                
                meta = outcomes_meta.get(pred_label, {'color': '#7e57c2', 'icon': '🔮', 'desc': 'Model predicted state.'})
                bg_color = meta['color']
                text_color = meta.get('color_text', '#ffffff')
                icon = meta['icon']
                desc = meta['desc']
                
                pred_prob = pred_probs[pred_idx] * 100
                
                st.markdown(f"""
                <div class="prediction-card" style="background-color: {bg_color}; color: {text_color};">
                    <span style="font-size: 4rem;">{icon}</span>
                    <h2 style="margin-top: 10px; font-weight: 800; color: {text_color};">{pred_label.upper()}</h2>
                    <p style="font-size: 1.15rem; max-width: 600px; margin: 10px auto; opacity: 0.9;">{desc}</p>
                    <div style="font-weight: 600; margin-top: 15px; font-size: 1.1rem;">
                        Prediction Confidence: {pred_prob:.2f}%
                    </div>
                    <div class="prob-bar-container" style="max-width: 300px; margin: 10px auto;">
                        <div class="prob-bar-fill" style="width: {pred_prob}%; background-color: {text_color};"></div>
                    </div>
                    <p style="font-size: 0.8rem; opacity: 0.7; max-width: 500px; margin: 5px auto;">
                        Note: Due to the uniform distribution of the dataset targets, individual probabilities cluster near the 10% random baseline.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show full distribution bar chart
                st.markdown("### 📊 Complete Match State Probabilities")
                prob_df = pd.DataFrame({
                    'Match Outcome': le.classes_,
                    'Probability (%)': pred_probs * 100
                }).sort_values('Probability (%)', ascending=False)
                
                st.bar_chart(prob_df, x='Match Outcome', y='Probability (%)', color='#ff2a5f')
                
            except Exception as transform_err:
                st.error("❌ Transformation Pipeline Mismatch!")
                st.markdown(
                    f"""
                    The feature extraction pipeline failed to align inputs to the LightGBM expected feature space.
                    
                    **Error Diagnostics:**
                    `{transform_err}`
                    """
                )


# ==========================================
# VIEW 2: 📊 BEHAVIORAL CONTRAST ENGINE
# ==========================================
elif navigation == "📊 Behavioral Contrast Engine":
    st.markdown('<div class="gradient-header">Behavioral Contrast Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-subtitle">Slicing dating segments and analyzing communication patterns</div>', unsafe_allow_html=True)
    
    # Top-level multi-select filter widgets
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🔍 Segment Selection Filters</div>', unsafe_allow_html=True)
    
    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        gender_filter = st.multiselect(
            "Filter by Gender", 
            options=sorted(df_raw['gender'].unique()), 
            default=df_raw['gender'].unique()
        )
    with fcol2:
        intent_filter = st.multiselect(
            "Filter by Relationship Intent", 
            options=sorted(df_raw['relationship_intent'].unique()), 
            default=df_raw['relationship_intent'].unique()
        )
    with fcol3:
        orientation_filter = st.multiselect(
            "Filter by Sexual Orientation", 
            options=sorted(df_raw['sexual_orientation'].unique()), 
            default=df_raw['sexual_orientation'].unique()
        )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mandatory outcome select widget directly above the charts
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Compare Match Outcomes</div>', unsafe_allow_html=True)
    
    match_options = sorted(df_raw['match_outcome'].dropna().unique().tolist())
    selected_outcomes = st.multiselect(
        "Select Match Outcomes to Compare", 
        options=match_options, 
        default=["Mutual Match", "Ghosted"]
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not selected_outcomes:
        st.warning("⚠️ Please select at least one Match Outcome to display the charts.")
    else:
        # Filter dataset based on both segment selection and match outcomes
        filtered_df = df_raw[
            df_raw['gender'].isin(gender_filter) &
            df_raw['relationship_intent'].isin(intent_filter) &
            df_raw['sexual_orientation'].isin(orientation_filter) &
            df_raw['match_outcome'].isin(selected_outcomes)
        ]
        
        if filtered_df.empty:
            st.warning("⚠️ No profiles match the chosen combination of filters. Please expand your criteria.")
        else:
            # Segment Metrics summary
            st.markdown("### 📈 Segment Overview Summary")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Matching Profiles", len(filtered_df))
            with mcol2:
                st.metric("Avg app usage (min)", f"{filtered_df['app_usage_time_min'].mean():.1f}")
            with mcol3:
                st.metric("Avg Swipe Right %", f"{filtered_df['swipe_right_ratio'].mean()*100:.1f}%")
            with mcol4:
                top_state = filtered_df['match_outcome'].mode()[0]
                st.metric("Top Match State", top_state)
            
            # Interactive Guide Expander
            with st.expander("💡 How to Discover Insights (Interactive Guide)", expanded=True):
                st.markdown(
                    """
                    Welcome to the Behavioral Contrast Engine! Use the interactive controls to discover patterns within user segments:
                    - **Filter the Noise:** Use the outcome filter widget above to compare exactly two contrasting outcomes (e.g., 'Mutual Match' vs. 'Ghosted').
                    - **Isolate Clusters:** Use the chart's native toolbar (Box Select or Lasso) to highlight and isolate specific dense user groups.
                    - **Hover for Context:** Hover over individual data points to reveal the exact user telemetry driving that outcome.
                    """
                )
                st.info(
                    "**🔍 Try This Scenario: The 'Power Swiper' Paradox**\n\n"
                    "Filter the chart to show ONLY **'Mutual Match'** and **'Ghosted'**. Set your axes to compare `swipe_right_ratio` against `app_usage_time_min`.\n\n"
                    "Look at the extreme top-right (users who spend hours on the app swiping right on everyone). Do these 'power users' actually secure more Mutual Matches, or does indiscriminate swiping lead straight to being Ghosted? Use the box-select tool to highlight this specific cluster and find out."
                )
                
            # Base color mapping configuration (reused across all plots)
            base_colors = [
                (255, 42, 95),   # Rose #ff2a5f
                (126, 87, 194),  # Violet #7e57c2
                (33, 150, 243),  # Blue #2196f3
                (76, 175, 80),   # Green #4caf50
                (255, 152, 0),   # Orange #ff9800
                (156, 39, 176),  # Purple #9c27b0
                (0, 188, 212),   # Cyan #00bcd4
                (233, 30, 99),   # Pink #e91e63
                (255, 235, 59),  # Yellow #ffeb3b
                (96, 125, 139)   # Blue Grey #607d8b
            ]
            
            # Prepare plotting dataframe with downsampling to keep Plotly responsive
            plot_df = filtered_df.copy()
            max_plot_rows = 3000
            if len(plot_df) > max_plot_rows:
                plot_df = plot_df.sample(max_plot_rows, random_state=42)
                
            outcome_mapping = {outcome: i for i, outcome in enumerate(selected_outcomes)}
            
            # Custom discrete legend HTML output (placed at the top of the charts)
            legend_html = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 25px; font-size: 0.95rem; justify-content: center;'>"
            for outcome, i in outcome_mapping.items():
                c = base_colors[i % len(base_colors)]
                hex_color = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
                legend_html += f"<div><span style='display:inline-block; width:12px; height:12px; background-color:{hex_color}; border-radius:50%; margin-right:5px; vertical-align: middle;'></span><strong style='vertical-align: middle;'>{outcome}</strong></div>"
            legend_html += "</div>"
            st.markdown(legend_html, unsafe_allow_html=True)
            
            import plotly.express as px
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">⚡ Interaction Dynamics: App Usage vs Swipe Right Ratio</div>', unsafe_allow_html=True)
                
                fig_scatter1 = px.scatter(
                    plot_df,
                    x='app_usage_time_min',
                    y='swipe_right_ratio',
                    color='match_outcome',
                    opacity=0.4,
                    labels={
                        'app_usage_time_min': 'App Usage Time (min)',
                        'swipe_right_ratio': 'Swipe Right Ratio',
                        'match_outcome': 'Match Outcome'
                    },
                    color_discrete_map={outcome: f"rgb({c[0]},{c[1]},{c[2]})" for outcome, i in outcome_mapping.items() for c in [base_colors[i % len(base_colors)]]}
                )
                fig_scatter1.update_layout(margin=dict(l=40, r=40, t=10, b=40), showlegend=False)
                st.plotly_chart(fig_scatter1, use_container_width=True)
                
                st.markdown("<p style='font-size:0.85rem; opacity:0.8;'>This chart maps swiping habits against app usage time. Marker opacity is set to 0.40 to reveal density overlap.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
            with chart_col2:
                st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                st.markdown('<div class="card-title">💬 Communication Patterns: Message Sent vs Emojis</div>', unsafe_allow_html=True)
                
                fig_scatter2 = px.scatter(
                    plot_df,
                    x='emoji_usage_rate',
                    y='message_sent_count',
                    color='match_outcome',
                    opacity=0.4,
                    labels={
                        'emoji_usage_rate': 'Emoji Usage Rate',
                        'message_sent_count': 'Message Sent Count',
                        'match_outcome': 'Match Outcome'
                    },
                    color_discrete_map={outcome: f"rgb({c[0]},{c[1]},{c[2]})" for outcome, i in outcome_mapping.items() for c in [base_colors[i % len(base_colors)]]}
                )
                fig_scatter2.update_layout(margin=dict(l=40, r=40, t=10, b=40), showlegend=False)
                st.plotly_chart(fig_scatter2, use_container_width=True)
                
                st.markdown("<p style='font-size:0.85rem; opacity:0.8;'>This contrast evaluates active messaging rates compared to emoji saturation, using color-coded category filters.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)


# ==========================================
# VIEW 3: 📉 MODEL BENCHMARKS & METRICS
# ==========================================
elif navigation == "📉 Model Benchmarks & Metrics":
    st.markdown('<div class="gradient-header">Model Benchmarks & Critique</div>', unsafe_allow_html=True)
    st.markdown('<div class="serif-subtitle">Evaluating LightGBM configurations against experimental baselines</div>', unsafe_allow_html=True)
    
    # Leaderboard Table
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏆 Model Comparison Leaderboard</div>', unsafe_allow_html=True)
    
    leaderboard_data = {
        'Model Framework': ['LightGBM (Tuned)', 'Logistic Regression', 'Extra Trees', 'Random Forest', 'Gradient Boosting', 'XGBoost', 'auto-sklearn baseline'],
        'Best F1 (Macro)': [0.0997, 0.0993, 0.0991, 0.0982, 0.0979, 0.0961, 0.1000],
        'Test Accuracy': [0.0998, 0.1025, 0.0991, 0.0987, 0.0991, 0.0961, 0.1012],
        'Tuning Time (s)': [172.93, 33.12, 135.66, 264.68, 1096.95, 121.27, 3600.0]
    }
    leaderboard_df = pd.DataFrame(leaderboard_data)
    st.dataframe(
        leaderboard_df.style.highlight_max(subset=['Best F1 (Macro)', 'Test Accuracy'], color='rgba(255, 42, 95, 0.15)'),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Critique Callout Block
    st.markdown("""
    <div class="critique-box">
        <h4 style="margin-top:0; color:#7e57c2; font-weight:800; font-size:1.25rem; border-bottom: 2px solid rgba(126, 87, 194, 0.2); padding-bottom: 5px;">⚠️ Academic Critique & Model Evaluation FAQ</h4>
        <div style="margin-top: 15px; font-size: 0.95rem; line-height: 1.6;">
            <strong>1. What is the best evaluation method for our models?</strong><br>
            Since the dataset contains 10 target classes that are perfectly balanced (each class representing exactly 10% of the dataset), <strong>Classification Accuracy</strong> and the <strong>Macro-averaged F1-Score</strong> are the best evaluation metrics. The <strong>Macro F1-score</strong> is particularly crucial here because it treats all classes equally and calculates the average F1-score across all 10 outcomes. This prevents any localized model failure or extreme bias from being hidden by overall metrics.
        </div>
        <div style="margin-top: 15px; font-size: 0.95rem; line-height: 1.6;">
            <strong>2. How do we compare our model to other models?</strong><br>
            We split the dataset using a Stratified Train-Test split (70% train, 30% test) to ensure class parity across splits. We then evaluate and compare the models based on:
            <ul style="margin-top: 5px; margin-bottom: 5px; padding-left: 20px;">
                <li><strong>Macro F1 & Accuracy:</strong> Evaluating overall predictive power compared to the 10% random baseline.</li>
                <li><strong>Confusion Matrix Analysis:</strong> Plotting predicted versus true label distributions to verify if the model learns any structured pattern.</li>
                <li><strong>Tuning Efficiency:</strong> Comparing the runtime duration required to find optimal parameters.</li>
            </ul>
        </div>
        <div style="margin-top: 15px; font-size: 0.95rem; line-height: 1.6;">
            <strong>3. How does our custom model compare to auto-sklearn?</strong><br>
            <ul style="margin-top: 5px; margin-bottom: 5px; padding-left: 20px;">
                <li>Our tuned <strong>LightGBM</strong> classifier achieves a Macro F1 of <strong>0.0997</strong> and a test accuracy of <strong>0.0998</strong> (tuning time: 172.93s).</li>
                <li>The <strong>auto-sklearn</strong> baseline (trained with a 1-hour time budget) achieves a validation Macro F1 of <strong>0.1000</strong> and a test accuracy of <strong>0.1012</strong>.</li>
            </ul>
            <strong>Verdict:</strong> Both our custom pipeline and the auto-sklearn benchmark converge exactly at the <strong>10% random baseline</strong>. This proves that no ensembling or automated framework can extract signal from a target vector composed of uniform random noise, validating that the target is mathematically independent of the features.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Importance details
    st.markdown("### 🔑 LightGBM Feature Importance Summary")
    
    col_feat1, col_feat2 = st.columns([1, 1])
    
    with col_feat1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Most Influential Split Features</div>', unsafe_allow_html=True)
        
        # Hardcoded from model inspection
        imp_data = {
            'Feature Name': [
                'body_type_Slim', 'sexual_orientation_Queer', 'location_type_Metro', 
                'gender_Male', 'swipe_time_of_day_Evening', 'swipe_time_of_day_Afternoon',
                'relationship_intent_Networking', 'swipe_time_of_day_Morning', 'body_type_Muscular',
                'sexual_orientation_Gay'
            ],
            'Split Count Importance': [79548, 62603, 54819, 51046, 40217, 3982, 3982, 3954, 3918, 3905]
        }
        st.bar_chart(pd.DataFrame(imp_data), x='Feature Name', y='Split Count Importance', color='#7e57c2')
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_feat2:
        st.markdown('<div class="premium-card" style="height:100%;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Observations on Splits</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - **Feature Spikes:** The five features `body_type_Slim`, `sexual_orientation_Queer`, `location_type_Metro`, `gender_Male`, and `swipe_time_of_day_Evening` dominate the splits by several orders of magnitude.
            - **Tree Structure Cues:** This indicates that the LightGBM trees split repeatedly on these nominal dummy categories trying to partition the uniform noise. In a dataset where classes are balanced and features contain no correlation to the target, the algorithm isolates random local variance clusters to minimize impurity, resulting in overfitting to the training noise.
            - **RobustScaler Impact:** Robust scaling ensures that continuous metrics such as `age` and `mutual_matches` do not scale-skew the splits, centering features safely away from outliers.
            """
        )
        st.markdown('</div>', unsafe_allow_html=True)
