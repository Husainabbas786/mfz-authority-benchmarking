import streamlit as st
import pandas as pd
import os

# Data last updated date - change this when you update the database
LAST_UPDATED = "2024-12-30"

# Initialize session state for recent searches
if 'recent_bulk_searches' not in st.session_state:
    st.session_state.recent_bulk_searches = []

# Initialize session state for filtered activities (shared across tabs)
if 'filtered_activities' not in st.session_state:
    st.session_state.filtered_activities = []

# Initialize session state for report popups
if 'show_over_reg' not in st.session_state:
    st.session_state.show_over_reg = False
if 'show_gap' not in st.session_state:
    st.session_state.show_gap = False

# Page config
st.set_page_config(
    page_title="MFZ Authority Approval Benchmarking",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics and animations
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 1.5rem;
        animation: fadeInDown 0.6s ease-out;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
        animation: fadeInDown 0.6s ease-out 0.1s both;
    }
    
    /* Search container */
    .search-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Cards */
    .comparison-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        animation: fadeInUp 0.5s ease-out;
        border: 1px solid #e5e7eb;
    }
    
    .comparison-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-same {
        background: #dbeafe;
        color: #1e40af;
    }
    
    .badge-over {
        background: #dcfce7;
        color: #166534;
    }
    
    .badge-under {
        background: #fef3c7;
        color: #92400e;
    }
    
    .badge-inactive {
        background: #f3f4f6;
        color: #6b7280;
        font-size: 0.65rem;
        padding: 0.2rem 0.5rem;
        margin-left: 0.5rem;
    }
    
    .badge-stop {
        background: #fee2e2;
        color: #991b1b;
        font-size: 0.65rem;
        padding: 0.2rem 0.5rem;
        margin-left: 0.5rem;
    }
    
    .badge-warning {
        background: #fef3c7;
        color: #92400e;
        font-size: 0.65rem;
        padding: 0.2rem 0.5rem;
        margin-left: 0.5rem;
    }
    
    /* MFZ Panel */
    .mfz-panel {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        border-radius: 16px;
        padding: 2rem;
        animation: fadeInLeft 0.5s ease-out;
    }
    
    .mfz-panel h3 {
        color: white;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    
    .mfz-detail {
        margin-bottom: 1rem;
    }
    
    .mfz-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.25rem;
    }
    
    .mfz-value {
        font-size: 1rem;
        color: white;
        font-weight: 500;
    }
    
    /* Insight bar */
    .insight-bar {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0;
        animation: fadeIn 0.5s ease-out 0.3s both;
    }
    
    .insight-text {
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        animation: fadeIn 0.5s ease-out;
    }
    
    .styled-table th {
        background: #f9fafb;
        padding: 1rem;
        text-align: left;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #e5e7eb;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .styled-table td {
        padding: 1rem;
        border-bottom: 1px solid #f3f4f6;
        color: #4b5563;
        transition: background 0.2s ease;
    }
    
    .styled-table tr:hover td {
        background: #f9fafb;
    }
    
    /* Match score bar */
    .score-bar {
        width: 100%;
        height: 6px;
        background: #e5e7eb;
        border-radius: 3px;
        overflow: hidden;
    }
    
    .score-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
        transition: width 0.5s ease-out;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 1px solid #e5e7eb;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 0;
        font-weight: 500;
        color: #6b7280;
    }
    
    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom-color: #667eea;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        border-color: #e5e7eb;
    }
    
    /* Allow full text in selectbox dropdown */
    .stSelectbox [data-baseweb="select"] span {
        white-space: normal !important;
    }
    
    /* Dropdown menu items - wrap text */
    div[data-baseweb="popover"] li {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.5 !important;
        padding: 12px 16px !important;
        min-height: auto !important;
    }
    
    div[data-baseweb="popover"] ul {
        max-height: 500px !important;
    }
    
    /* Multiselect tags wrap */
    .stMultiSelect [data-baseweb="tag"] {
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        border-radius: 8px;
    }
    
    /* Recent search buttons */
    .stButton > button[kind="secondary"] {
        background: #f8fafc;
        color: #374151;
        border: 1px solid #e5e7eb;
        font-size: 0.75rem;
        padding: 0.5rem;
        text-align: left;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f1f5f9;
        border-color: #667eea;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #374151;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: #e5e7eb;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Use relative paths for deployment
    df_all = pd.read_excel('All_FreeZones_Standardized.xlsx')
    df_matches_mfz = pd.read_excel('Activity_Matches_Semantic.xlsx')
    df_matches_det = pd.read_csv('Activity_Matches_DET.csv.gz', compression='gzip')
    
    # Add Status column to MFZ matches if not present
    if 'Competitor_Status' not in df_matches_mfz.columns:
        status_map = df_all.set_index(['Free_Zone', 'Activity_Name'])['Status'].to_dict()
        df_matches_mfz['Competitor_Status'] = df_matches_mfz.apply(
            lambda row: status_map.get((row['Competitor_FZ'], row['Competitor_Name']), 'Active'), axis=1
        )
    
    # Standardize DET matches column names to match MFZ format
    df_matches_det = df_matches_det.rename(columns={
        'Base_Code': 'MFZ_Code',
        'Base_Name': 'MFZ_Name',
        'Base_Category': 'MFZ_Category',
        'Base_Requires_Approval': 'MFZ_Requires_Approval',
        'Base_Authority': 'MFZ_Authority',
        'Base_Status': 'MFZ_Status',
        'Base_Over_Regulating': 'MFZ_Over_Regulating',
        'Base_Under_Regulating': 'MFZ_Under_Regulating',
    })
    
    return df_all, df_matches_mfz, df_matches_det

df_all, df_matches_mfz, df_matches_det = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### Universe")
    selected_universe = st.selectbox(
        "Base Free Zone",
        options=["MFZ", "DET"],
        index=0,
        help="Select which free zone to use as the base for comparison"
    )
    
    # Set data based on universe
    if selected_universe == "MFZ":
        df_matches = df_matches_mfz
        base_fz = "MFZ"
    else:
        df_matches = df_matches_det
        base_fz = "DET"
    
    # Get base activities for search (Code first, then name)
    base_df = df_all[df_all['Free_Zone'] == base_fz][['Activity_Code', 'Activity_Name']].drop_duplicates()
    base_activities = [f"{row['Activity_Code']} - {row['Activity_Name']}" for _, row in base_df.iterrows()]
    base_activities.sort()
    
    st.markdown("---")
    st.markdown("### Filters")
    
    # Authority filter
    all_authorities = df_matches['MFZ_Authority'].dropna().unique().tolist()
    all_authorities = [a for a in all_authorities if str(a).strip() and str(a) != 'nan']
    all_authorities = list(set([item.strip() for sublist in [str(a).split(',') for a in all_authorities] for item in sublist if item.strip()]))
    all_authorities.sort()
    
    selected_authorities = st.multiselect(
        "Filter by Authority",
        options=all_authorities,
        default=[],
        help="Show only activities requiring specific authorities"
    )
    
    # Free zone filter
    competitor_fzs = df_matches['Competitor_FZ'].unique().tolist()
    selected_fzs = st.multiselect(
        "Filter by Competitor",
        options=competitor_fzs,
        default=competitor_fzs,
        help="Select competitors to compare"
    )
    
    # Activity Code filter (searchable)
    all_codes = df_matches['MFZ_Code'].unique().tolist()
    all_codes.sort()
    selected_codes = st.multiselect(
        f"Filter by {base_fz} Code",
        options=all_codes,
        default=[],
        help="Type to search codes (leave empty for all)"
    )
    
    # Match score threshold
    min_score = st.slider(
        "Minimum Match Score",
        min_value=0.50,
        max_value=1.0,
        value=0.80,
        step=0.05,
        help="Higher score = more confident match"
    )
    
    st.markdown("---")
    
    # Overview Metrics
    st.markdown("### Overview Metrics")
    st.metric("Total Activities", f"{len(df_all):,}")
    st.metric("Matched Pairs", f"{len(df_matches):,}")
    st.metric(f"{base_fz} Matched", f"{df_matches['MFZ_Code'].nunique():,}")
    st.metric("Avg Match Score", f"{df_matches['Match_Score'].mean()*100:.0f}%")
    
    st.markdown("---")
    
    st.markdown("### Regulation Reports")
    
    # Over-Regulation Report Button
    if st.button("📋 Over-Regulation Report", use_container_width=True):
        st.session_state.show_over_reg = True
        st.session_state.show_gap = False
    
    # Gap Analysis Report Button
    if st.button("📋 Gap Analysis Report", use_container_width=True):
        st.session_state.show_gap = True
        st.session_state.show_over_reg = False

# Main content
st.markdown('<p class="main-header">Authority Approval Benchmarking</p>', unsafe_allow_html=True)

# Over-Regulation Report Popup
if st.session_state.show_over_reg:
    st.markdown("---")
    col_title, col_close = st.columns([6, 1])
    with col_title:
        st.markdown(f"#### 📋 Over-Regulation Report ({base_fz})")
        st.markdown(f"Activities where {base_fz} requires approval but competitors do not")
    with col_close:
        if st.button("✕ Close", key="close_over_reg"):
            st.session_state.show_over_reg = False
            st.rerun()
    
    over_reg = df_matches[df_matches['MFZ_Over_Regulating'] == True].copy()
    
    # Apply sidebar filters
    if selected_authorities:
        over_reg = over_reg[over_reg['MFZ_Authority'].apply(lambda x: any(auth in str(x) for auth in selected_authorities))]
    if selected_fzs:
        over_reg = over_reg[over_reg['Competitor_FZ'].isin(selected_fzs)]
    over_reg = over_reg[over_reg['Match_Score'] >= min_score]
    
    if len(over_reg) > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**By Authority**")
            auth_counts = over_reg['MFZ_Authority'].value_counts().head(10)
            st.dataframe(auth_counts, use_container_width=True)
        with col2:
            st.bar_chart(auth_counts)
        
        display_cols = ['MFZ_Name', 'MFZ_Authority', 'Competitor_FZ', 'Competitor_Name', 'Match_Score']
        over_reg_display = over_reg[display_cols].copy()
        over_reg_display.columns = [f'{base_fz} Activity', f'{base_fz} Authority', 'Competitor', 'Competitor Activity', 'Match Score']
        over_reg_display['Match Score'] = (over_reg_display['Match Score'] * 100).round(0).astype(int).astype(str) + '%'
        
        st.dataframe(over_reg_display, use_container_width=True, hide_index=True, height=300)
        
        csv = over_reg.to_csv(index=False)
        st.download_button("Download Report", data=csv, file_name=f"{base_fz}_Over_Regulation.csv", mime="text/csv")
    else:
        st.info("No over-regulation cases found.")
    st.markdown("---")

# Gap Analysis Report Popup
if st.session_state.show_gap:
    st.markdown("---")
    col_title, col_close = st.columns([6, 1])
    with col_title:
        st.markdown(f"#### 📋 Gap Analysis Report ({base_fz})")
        st.markdown(f"Activities where competitors require approval but {base_fz} does not")
    with col_close:
        if st.button("✕ Close", key="close_gap"):
            st.session_state.show_gap = False
            st.rerun()
    
    under_reg = df_matches[df_matches['MFZ_Under_Regulating'] == True].copy()
    
    # Apply sidebar filters
    if selected_fzs:
        under_reg = under_reg[under_reg['Competitor_FZ'].isin(selected_fzs)]
    under_reg = under_reg[under_reg['Match_Score'] >= min_score]
    
    if len(under_reg) > 0:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**By Competitor Authority**")
            comp_auth_counts = under_reg['Competitor_Authority'].value_counts().head(10)
            st.dataframe(comp_auth_counts, use_container_width=True)
        with col2:
            st.bar_chart(comp_auth_counts)
        
        display_cols = ['MFZ_Name', 'Competitor_FZ', 'Competitor_Name', 'Competitor_Authority', 'Match_Score']
        under_reg_display = under_reg[display_cols].copy()
        under_reg_display.columns = [f'{base_fz} Activity', 'Competitor', 'Competitor Activity', 'Required Authority', 'Match Score']
        under_reg_display['Match Score'] = (under_reg_display['Match Score'] * 100).round(0).astype(int).astype(str) + '%'
        
        st.dataframe(under_reg_display, use_container_width=True, hide_index=True, height=300)
        
        csv = under_reg.to_csv(index=False)
        st.download_button("Download Report", data=csv, file_name=f"{base_fz}_Gap_Analysis.csv", mime="text/csv")
    else:
        st.info("No under-regulation cases (gaps) found.")
    st.markdown("---")

# Tabs
tab1, tab4 = st.tabs(["Activity Search", "Bulk Compare"])

# Tab 1: Activity Search
with tab1:
    if 'search_key' not in st.session_state:
        st.session_state.search_key = 0
    
    # Sort activities by code ascending
    def sort_by_code(activity):
        code = activity.split(' - ')[0].strip()
        try:
            if '.' in code:
                parts = code.split('.')
                return (int(float(parts[0])), int(float(parts[1])) if len(parts) > 1 else 0)
            else:
                return (int(float(code)), 0)
        except:
            return (999999, 0)
    
    sorted_activities = sorted(base_activities, key=sort_by_code)
    
    # Search input
    col_search, col_clear = st.columns([10, 1])
    
    with col_search:
        search_query = st.text_input(
            f"Search {base_fz} Activity",
            value="",
            placeholder="Type code (e.g., 4790) or name (e.g., e-commerce)...",
            key=f"search_input_{st.session_state.search_key}"
        )
    
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕", key="clear_search", help="Clear"):
            st.session_state.search_key += 1
            st.rerun()
    
    # Filter logic
    if search_query:
        query = search_query.strip()
        
        # Check if code search (starts with digit)
        if query[0].isdigit():
            # CODE SEARCH: only codes STARTING with query (exact prefix match)
            filtered = [a for a in sorted_activities if a.split(' - ')[0].startswith(query)]
        else:
            # NAME SEARCH: normalize query and names for flexible matching
            # Remove special chars and spaces for comparison
            query_normalized = query.lower().replace('-', '').replace(' ', '')
            filtered = []
            for a in sorted_activities:
                name = a.split(' - ', 1)[1] if ' - ' in a else a
                # Normalize name the same way
                name_normalized = name.lower().replace('-', '').replace(' ', '')
                # Also check individual words
                words = name.lower().replace('-', ' ').replace('/', ' ').split()
                
                # Match if: normalized name contains normalized query, OR any word starts with query
                if query_normalized in name_normalized or any(w.startswith(query.lower()) for w in words):
                    filtered.append(a)
        
        # Show results
        if filtered:
            st.caption(f"Found {len(filtered)} activities")
            selected_activity = st.selectbox(
                "Select activity",
                options=[""] + filtered,
                index=0,
                label_visibility="collapsed",
                key=f"select_{st.session_state.search_key}"
            )
        else:
            st.warning(f"No activities found for '{query}'")
            selected_activity = ""
    else:
        # No search - show all
        selected_activity = st.selectbox(
            "Select activity",
            options=[""] + sorted_activities,
            index=0,
            label_visibility="collapsed",
            key=f"select_{st.session_state.search_key}"
        )
    
    # Rest of Tab 1 - display selected activity
    if selected_activity:
        activity_name = selected_activity.split(' - ', 1)[1] if ' - ' in selected_activity else selected_activity
        st.session_state.filtered_activities = [activity_name]
        
        mfz_data = df_all[(df_all['Free_Zone'] == base_fz) & (df_all['Activity_Name'] == activity_name)].iloc[0]
        matches = df_matches[df_matches['MFZ_Name'] == activity_name].copy()
        
        if selected_authorities:
            matches = matches[matches['MFZ_Authority'].apply(lambda x: any(auth in str(x) for auth in selected_authorities))]
        if selected_fzs:
            matches = matches[matches['Competitor_FZ'].isin(selected_fzs)]
        if selected_codes:
            matches = matches[matches['MFZ_Code'].isin(selected_codes)]
        matches = matches[matches['Match_Score'] >= min_score]
        matches = matches.sort_values('Match_Score', ascending=False)
        
        col_mfz, col_comp = st.columns([1, 2.5])
        
        with col_mfz:
            st.markdown(f"""
            <div class="mfz-panel">
                <h3>{base_fz} Activity</h3>
                <div class="mfz-detail">
                    <div class="mfz-label">Activity Name</div>
                    <div class="mfz-value">{mfz_data['Activity_Name'][:50]}</div>
                    <div style="font-size: 0.8rem; color: #9ca3af; margin-top: 0.25rem;">Code: {mfz_data['Activity_Code']}</div>
                </div>
                <div class="mfz-detail">
                    <div class="mfz-label">Category</div>
                    <div class="mfz-value">{mfz_data['Category'] if pd.notna(mfz_data['Category']) else 'N/A'}</div>
                </div>
                <div class="mfz-detail">
                    <div class="mfz-label">Requires Approval</div>
                    <div class="mfz-value">{'Yes' if mfz_data['Requires_Approval'] else 'No'}</div>
                </div>
                <div class="mfz-detail">
                    <div class="mfz-label">Authority</div>
                    <div class="mfz-value">{mfz_data['Authority_Code'] if pd.notna(mfz_data['Authority_Code']) else 'None'}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_comp:
            if len(matches) > 0:
                st.markdown("#### Competitor Comparison")
                
                same_count = len(matches[matches['Same_Status'] == True])
                over_count = len(matches[matches['MFZ_Over_Regulating'] == True])
                under_count = len(matches[matches['MFZ_Under_Regulating'] == True])
                total_matches = len(matches)
                
                if over_count > under_count:
                    insight = f"{base_fz} requires MORE approvals than {over_count} of {total_matches} matched competitors."
                elif under_count > over_count:
                    insight = f"{base_fz} requires FEWER approvals than {under_count} of {total_matches} matched competitors."
                else:
                    insight = f"{base_fz} approval requirements are aligned with {same_count} of {total_matches} matched competitors."
                
                st.markdown(f'<div class="insight-bar"><div class="insight-text">{insight}</div></div>', unsafe_allow_html=True)
                
                card_cols = st.columns(min(len(matches), 4))
                for idx, (_, match) in enumerate(matches.head(4).iterrows()):
                    with card_cols[idx]:
                        if match['MFZ_Over_Regulating']:
                            reg_status = f"{base_fz} Over-regulating"
                            badge_class = "badge-over"
                        elif match['MFZ_Under_Regulating']:
                            reg_status = f"{base_fz} Under-regulating"
                            badge_class = "badge-under"
                        else:
                            reg_status = "Same Requirements"
                            badge_class = "badge-same"
                        
                        comp_auth = match['Competitor_Authority'] if pd.notna(match['Competitor_Authority']) else 'None'
                        comp_code = match.get('Competitor_Code', 'N/A')
                        activity_status = match.get('Competitor_Status', 'Active')
                        status_badge = ''
                        if activity_status == 'InActive':
                            status_badge = '<span class="badge badge-inactive">Inactive</span>'
                        elif activity_status == 'Stop':
                            status_badge = '<span class="badge badge-stop">Stopped</span>'
                        elif activity_status == 'Warning':
                            status_badge = '<span class="badge badge-warning">Warning</span>'
                        
                        st.markdown(f"""
                        <div class="comparison-card">
                            <div style="font-weight: 600; color: #1a1a2e; margin-bottom: 0.5rem;">{match['Competitor_FZ']}{status_badge}</div>
                            <div style="font-size: 0.85rem; color: #6b7280; margin-bottom: 0.25rem;">{match['Competitor_Name'][:35]}...</div>
                            <div style="font-size: 0.75rem; color: #9ca3af; margin-bottom: 1rem;">Code: {comp_code}</div>
                            <div class="score-bar"><div class="score-fill" style="width: {match['Match_Score']*100}%"></div></div>
                            <div style="font-size: 0.75rem; color: #9ca3af; margin: 0.5rem 0;">{int(match['Match_Score']*100)}% match</div>
                            <div style="margin-bottom: 0.75rem;"><span class="badge {badge_class}">{reg_status}</span></div>
                            <div style="font-size: 0.8rem; color: #374151;"><strong>Authority:</strong> {comp_auth}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown(f"#### Detailed Comparison")
                st.caption(f"Data last updated: {LAST_UPDATED}")
                table_data = matches[['Competitor_FZ', 'Competitor_Code', 'Competitor_Name', 'Competitor_Status', 'Competitor_Requires_Approval', 'Competitor_Authority', 'Match_Score']].copy()
                table_data.columns = ['Free Zone', 'Code', 'Activity Name', 'Status', 'Requires Approval', 'Authority', 'Match Score']
                table_data['Requires Approval'] = table_data['Requires Approval'].map({True: 'Yes', False: 'No'})
                table_data['Match Score'] = (table_data['Match Score'] * 100).round(0).astype(int).astype(str) + '%'
                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.info("No matches found for this activity with current filters.")

# Tab 4: Bulk Compare
with tab4:
    st.markdown("#### Bulk Compare")
    st.markdown(f"Compare multiple {base_fz} activities at once")
    
    # Recent searches section
    if st.session_state.recent_bulk_searches:
        st.markdown("##### Recent Searches")
        
        # Display recent searches as clickable buttons
        cols = st.columns(min(len(st.session_state.recent_bulk_searches), 3))
        for idx, search in enumerate(st.session_state.recent_bulk_searches[:9]):  # Show max 9
            col_idx = idx % 3
            with cols[col_idx]:
                search_label = f"{len(search)} activities"
                search_preview = ", ".join([s.split(' - ')[1][:20] if ' - ' in s else s[:20] for s in search[:2]])
                if len(search) > 2:
                    search_preview += "..."
                if st.button(f"{search_label}\n{search_preview}", key=f"recent_{idx}", use_container_width=True):
                    st.session_state.selected_bulk = search
        
        st.markdown("---")
    
    # File upload section
    st.markdown("##### Upload Activities from Excel")
    uploaded_file = st.file_uploader(
        "Upload Excel file with Code and Activity Name columns",
        type=['xlsx', 'xls'],
        help="Excel must have 'Code' and 'Activity Name' columns"
    )
    
    uploaded_activities = []
    if uploaded_file is not None:
        try:
            df_upload = pd.read_excel(uploaded_file)
            # Clean column names (remove extra spaces)
            df_upload.columns = df_upload.columns.str.strip()
            
            # Find code and name columns (flexible matching)
            code_col = None
            name_col = None
            for col in df_upload.columns:
                if 'code' in col.lower():
                    code_col = col
                if 'name' in col.lower() or 'activity' in col.lower():
                    name_col = col
            
            if code_col and name_col:
                # Extract activities
                for _, row in df_upload.iterrows():
                    code = str(row[code_col]).strip()
                    name = str(row[name_col]).strip()
                    # Find matching activity in base_activities
                    for ba in base_activities:
                        if code in ba or name.lower() in ba.lower():
                            if ba not in uploaded_activities:
                                uploaded_activities.append(ba)
                            break
                
                st.success(f"Found {len(uploaded_activities)} matching activities from uploaded file")
            else:
                st.error("Could not find 'Code' and 'Activity Name' columns. Please check your file format.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("##### Or Select Activities Manually")
    
    # Check if we have a selection from recent search or uploaded file
    default_selection = st.session_state.get('selected_bulk', [])
    if uploaded_activities:
        default_selection = uploaded_activities
    
    # Multi-select activities
    selected_activities = st.multiselect(
        f"Select {base_fz} Activities",
        options=base_activities,
        default=default_selection if default_selection else [],
        help="Select multiple activities to compare"
    )
    
    # Clear selected_bulk after using it so it doesn't keep reapplying
    if 'selected_bulk' in st.session_state and default_selection:
        st.session_state.selected_bulk = []
    
    # Save search button
    col1, col2 = st.columns([3, 1])
    with col2:
        if selected_activities and st.button("Save Search", use_container_width=True):
            # Add to recent searches (avoid duplicates)
            if selected_activities not in st.session_state.recent_bulk_searches:
                st.session_state.recent_bulk_searches.insert(0, selected_activities)
                # Keep only last 15 searches
                st.session_state.recent_bulk_searches = st.session_state.recent_bulk_searches[:15]
                st.success("Search saved!")
                st.rerun()
    
    # Clear recent searches
    with col1:
        if st.session_state.recent_bulk_searches:
            if st.button("Clear Recent Searches"):
                st.session_state.recent_bulk_searches = []
                st.session_state.selected_bulk = []
                st.rerun()
    
    if selected_activities:
        # Extract activity names (code is first)
        activity_names = [act.split(' - ', 1)[1] if ' - ' in act else act for act in selected_activities]
        
        # Always update session state for other tabs
        st.session_state.filtered_activities = activity_names
        
        bulk_matches = df_matches[df_matches['MFZ_Name'].isin(activity_names)].copy()
        
        # Apply filters
        if selected_fzs:
            bulk_matches = bulk_matches[bulk_matches['Competitor_FZ'].isin(selected_fzs)]
        bulk_matches = bulk_matches[bulk_matches['Match_Score'] >= min_score]
        
        # Summary metrics
        st.markdown("##### Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Activities Selected", len(activity_names))
        with col2:
            st.metric("Total Matches", len(bulk_matches))
        with col3:
            st.metric("Over-regulating", len(bulk_matches[bulk_matches['MFZ_Over_Regulating'] == True]))
        with col4:
            st.metric("Under-regulating", len(bulk_matches[bulk_matches['MFZ_Under_Regulating'] == True]))
        
        st.markdown("---")
        
        # Get unique competitors with matches
        competitors_with_matches = bulk_matches['Competitor_FZ'].unique().tolist()
        competitors_with_matches.sort()
        
        st.markdown("##### Results by Competitor")
        st.caption(f"Data last updated: {LAST_UPDATED}")
        
        # Create competitor cards in grid (2 per row)
        for i in range(0, len(competitors_with_matches), 2):
            cols = st.columns(2)
            
            for j, col in enumerate(cols):
                if i + j < len(competitors_with_matches):
                    competitor = competitors_with_matches[i + j]
                    comp_data = bulk_matches[bulk_matches['Competitor_FZ'] == competitor].copy()
                    
                    over_count = len(comp_data[comp_data['MFZ_Over_Regulating'] == True])
                    under_count = len(comp_data[comp_data['MFZ_Under_Regulating'] == True])
                    same_count = len(comp_data[comp_data['Same_Status'] == True])
                    
                    with col:
                        # Card header
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 1rem 1.5rem; 
                                    border-radius: 12px 12px 0 0; 
                                    color: white;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span style="font-size: 1.25rem; font-weight: 600;">{competitor}</span>
                                    <span style="font-size: 0.85rem; margin-left: 0.5rem; opacity: 0.9;">({len(comp_data)} matches)</span>
                                </div>
                            </div>
                            <div style="margin-top: 0.5rem; font-size: 0.75rem; opacity: 0.9;">
                                <span style="margin-right: 1rem;">Same: {same_count}</span>
                                <span style="margin-right: 1rem;">Over: {over_count}</span>
                                <span>Under: {under_count}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Table
                        table_cols = ['MFZ_Code', 'MFZ_Name', 'Competitor_Code', 'Competitor_Name', 'Competitor_Authority', 'Match_Score']
                        comp_display = comp_data[table_cols].copy()
                        comp_display.columns = [f'{base_fz} Code', f'{base_fz} Activity', 'Comp Code', 'Competitor Activity', 'Authority', 'Match']
                        comp_display['Match'] = (comp_display['Match'] * 100).round(0).astype(int).astype(str) + '%'
                        
                        st.dataframe(comp_display, use_container_width=True, hide_index=True, height=200)
                        
                        # Download button for this competitor
                        csv_comp = comp_data.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download {competitor}",
                            data=csv_comp,
                            file_name=f"{base_fz}_vs_{competitor}.csv",
                            mime="text/csv",
                            key=f"download_{competitor}"
                        )
                        
                        st.markdown("<br>", unsafe_allow_html=True)
        
        # Master download at the bottom
        st.markdown("---")
        st.markdown("##### Download All Data")
        csv_all = bulk_matches.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Report (All Competitors)",
            data=csv_all,
            file_name=f"{base_fz}_Bulk_Comparison_All.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Select activities above to begin bulk comparison")
