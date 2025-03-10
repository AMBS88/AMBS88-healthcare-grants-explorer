import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from institutional_profile import add_institutional_profile_to_app

# Set page configuration
st.set_page_config(
    page_title="Healthcare Grants Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Function to load grants data
@st.cache_data
def load_grants_data():
    try:
        # Try to load from the CSV file
        return pd.read_csv("data/healthcare_grants.csv")
    except FileNotFoundError:
        # If file doesn't exist, create sample data
        return create_sample_grants_data()

def create_sample_grants_data():
    """Create sample healthcare grants data"""
    grants_data = [
        {
            "title": "Innovative Immunotherapy Approaches for Cancer Treatment",
            "funding_agency": "National Institutes of Health",
            "description": "Research grant for developing novel immunotherapy approaches to treat cancer, particularly focusing on solid tumors resistant to current therapies.",
            "keywords": ["cancer", "immunotherapy", "clinical trial", "oncology", "resistance"],
            "award_amount": 2500000,
            "deadline": "2023-10-15",
            "relevant_departments": ["Department of Oncology", "Department of Immunology"],
            "url": "https://grants.nih.gov/sample-link"
        },
        {
            "title": "Cardiac Regeneration and Repair Mechanisms",
            "funding_agency": "American Heart Association",
            "description": "Research into cardiac tissue regeneration and repair mechanisms following heart damage, with a focus on stem cell approaches and tissue engineering.",
            "keywords": ["cardiology", "regenerative medicine", "stem cells", "heart failure", "tissue engineering"],
            "award_amount": 1800000,
            "deadline": "2023-11-30",
            "relevant_departments": ["Heart Institute", "Department of Regenerative Medicine"],
            "url": "https://professional.heart.org/sample-link"
        },
        {
            "title": "Advancing Neurological Rehabilitation Technology",
            "funding_agency": "National Science Foundation",
            "description": "Development and clinical validation of advanced technologies for neurological rehabilitation, including robotics, virtual reality, and AI-assisted therapy programs.",
            "keywords": ["rehabilitation", "neurology", "robotics", "stroke", "technology", "virtual reality"],
            "award_amount": 1500000,
            "deadline": "2023-12-15",
            "relevant_departments": ["Rehabilitation Center", "Neuroscience Center"],
            "url": "https://www.nsf.gov/sample-link"
        },
        {
            "title": "Pediatric Cancer Genomics and Precision Medicine",
            "funding_agency": "St. Baldrick's Foundation",
            "description": "Research into genomic drivers of pediatric cancers and development of precision medicine approaches for children with cancer.",
            "keywords": ["pediatric", "oncology", "genomics", "precision medicine", "children"],
            "award_amount": 1200000,
            "deadline": "2024-01-20",
            "relevant_departments": ["Pediatric Medical Center", "Department of Oncology"],
            "url": "https://www.stbaldricks.org/sample-link"
        },
        {
            "title": "Biomarkers for Neurodegenerative Disease Progression",
            "funding_agency": "Michael J. Fox Foundation",
            "description": "Identification and validation of biomarkers for tracking neurodegenerative disease progression and treatment response, with a focus on Parkinson's disease.",
            "keywords": ["neuroscience", "biomarkers", "neurodegenerative", "Parkinson's", "Alzheimer's"],
            "award_amount": 950000,
            "deadline": "2023-11-10",
            "relevant_departments": ["Neuroscience Center"],
            "url": "https://www.michaeljfox.org/sample-link"
        },
        {
            "title": "Medical Device Innovation for Heart Failure Management",
            "funding_agency": "Medical Technology Enterprise Consortium",
            "description": "Development of innovative medical devices for monitoring and managing heart failure in outpatient settings.",
            "keywords": ["cardiology", "medical device", "heart failure", "monitoring", "technology"],
            "award_amount": 1350000,
            "deadline": "2024-02-28",
            "relevant_departments": ["Heart Institute", "Department of Medical Engineering"],
            "url": "https://mtec-sc.org/sample-link"
        },
        {
            "title": "AI Applications in Medical Imaging Diagnostics",
            "funding_agency": "National Institutes of Health",
            "description": "Development and validation of artificial intelligence algorithms for improving medical imaging diagnostics across multiple specialties.",
            "keywords": ["AI", "imaging", "diagnostics", "radiology", "machine learning"],
            "award_amount": 1600000,
            "deadline": "2023-12-05",
            "relevant_departments": ["Department of Radiology", "Department of Computer Science"],
            "url": "https://grants.nih.gov/sample-link-2"
        },
        {
            "title": "Novel Therapeutic Approaches for Multiple Sclerosis",
            "funding_agency": "National Multiple Sclerosis Society",
            "description": "Research into novel therapeutic approaches for multiple sclerosis, focusing on neuroprotection and remyelination strategies.",
            "keywords": ["multiple sclerosis", "neurology", "immunology", "clinical trial", "neuroprotection"],
            "award_amount": 1100000,
            "deadline": "2024-01-15",
            "relevant_departments": ["Neuroscience Center", "Department of Immunology"],
            "url": "https://www.nationalmssociety.org/sample-link"
        },
        {
            "title": "Improving Outcomes in Pediatric Rare Diseases",
            "funding_agency": "NIH Rare Diseases Clinical Research Network",
            "description": "Clinical research to improve diagnosis, treatment, and outcomes for children with rare diseases.",
            "keywords": ["pediatric", "rare disease", "genetics", "clinical research", "outcome measures"],
            "award_amount": 1750000,
            "deadline": "2024-03-10",
            "relevant_departments": ["Pediatric Medical Center", "Department of Genetics"],
            "url": "https://ncats.nih.gov/rdcrn/sample-link"
        },
        {
            "title": "Cancer Immunotherapy Resistance Mechanisms",
            "funding_agency": "Cancer Research Institute",
            "description": "Research into mechanisms of resistance to cancer immunotherapy and strategies to overcome treatment resistance.",
            "keywords": ["cancer", "immunotherapy", "resistance", "oncology", "T cells"],
            "award_amount": 1400000,
            "deadline": "2023-11-25",
            "relevant_departments": ["Department of Oncology", "Department of Immunology"],
            "url": "https://www.cancerresearch.org/sample-link"
        },
        {
            "title": "Robotic Assistance in Surgical Procedures",
            "funding_agency": "Department of Defense",
            "description": "Development and testing of advanced robotic systems to assist surgeons in complex procedures, with a focus on trauma and battlefield applications.",
            "keywords": ["surgery", "robotics", "trauma", "technology", "minimally invasive"],
            "award_amount": 2200000,
            "deadline": "2024-02-15",
            "relevant_departments": ["Department of Surgery", "Department of Medical Engineering"],
            "url": "https://cdmrp.army.mil/sample-link"
        },
        {
            "title": "Virtual Reality Applications in Pain Management",
            "funding_agency": "Patient-Centered Outcomes Research Institute",
            "description": "Research into the effectiveness of virtual reality interventions for chronic pain management across different patient populations.",
            "keywords": ["pain management", "virtual reality", "patient outcomes", "rehabilitation", "chronic pain"],
            "award_amount": 850000,
            "deadline": "2024-01-30",
            "relevant_departments": ["Rehabilitation Center", "Department of Pain Medicine"],
            "url": "https://www.pcori.org/sample-link"
        }
    ]
    
    df = pd.DataFrame(grants_data)
    # Save the sample data to CSV
    df.to_csv("data/healthcare_grants.csv", index=False)
    return df

# Get the grants data
grants_df = load_grants_data()

# Convert the relevant_departments column from string to list if needed
if 'relevant_departments' in grants_df.columns and grants_df['relevant_departments'].dtype == 'object':
    grants_df['relevant_departments'] = grants_df['relevant_departments'].apply(
        lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
    )

# Convert the keywords column from string to list if needed
if 'keywords' in grants_df.columns and grants_df['keywords'].dtype == 'object':
    grants_df['keywords'] = grants_df['keywords'].apply(
        lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else x
    )

# Title and intro
st.title("Healthcare Grants Explorer")
st.markdown("---")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Grant Explorer", "Analytics", "Institutional Profile", "About"])

# =======================================================
# TAB 1: GRANT EXPLORER
# =======================================================
with tab1:
    st.header("Grant Explorer")
    
    # Create a search and filter sidebar
    st.sidebar.header("Search & Filter")
    
    # Search by keyword
    search_term = st.sidebar.text_input("Search by keyword")
    
    # Filter by funding agency
    funding_agencies = ["All"] + sorted(grants_df["funding_agency"].unique().tolist())
    selected_agency = st.sidebar.selectbox("Filter by Funding Agency", funding_agencies)
    
    # Filter by department
    all_departments = []
    for dept_list in grants_df["relevant_departments"].dropna():
        if isinstance(dept_list, list):
            all_departments.extend(dept_list)
    unique_departments = ["All"] + sorted(list(set(all_departments)))
    selected_dept = st.sidebar.selectbox("Filter by Department", unique_departments)
    
    # Filter by minimum award amount
    min_amount = st.sidebar.slider(
        "Minimum Award Amount",
        min_value=0,
        max_value=3000000,
        value=0,
        step=100000,
        format="$%d"
    )
    
    # Apply filters
    filtered_df = grants_df.copy()
    
    # Apply search term filter
    if search_term:
        mask = (
            filtered_df["title"].str.contains(search_term, case=False, na=False) |
            filtered_df["description"].str.contains(search_term, case=False, na=False) |
            filtered_df["keywords"].apply(
                lambda keywords: any(search_term.lower() in keyword.lower() for keyword in keywords) 
                if isinstance(keywords, list) else False
            )
        )
        filtered_df = filtered_df[mask]
    
    # Apply funding agency filter
    if selected_agency != "All":
        filtered_df = filtered_df[filtered_df["funding_agency"] == selected_agency]
    
    # Apply department filter
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df["relevant_departments"].apply(
            lambda depts: selected_dept in depts if isinstance(depts, list) else False
        )]
    
    # Apply award amount filter
    filtered_df = filtered_df[filtered_df["award_amount"] >= min_amount]
    
    # Display results
    st.subheader(f"Found {len(filtered_df)} Matching Grants")
    
    if len(filtered_df) > 0:
        # Format deadline column
        if 'deadline' in filtered_df.columns:
            filtered_df['deadline'] = pd.to_datetime(filtered_df['deadline']).dt.strftime('%Y-%m-%d')
        
        # Format award amount column
        if 'award_amount' in filtered_df.columns:
            filtered_df['award_amount'] = filtered_df['award_amount'].apply(lambda x: f"${x:,.2f}")
        
        # Display grants as expandable sections
        for i, row in filtered_df.iterrows():
            with st.expander(f"{row['title']} - {row['funding_agency']}"):
                st.write(f"**Description:** {row['description']}")
                st.write(f"**Award Amount:** {row['award_amount']}")
                st.write(f"**Deadline:** {row['deadline']}")
                
                # Display keywords
                if 'keywords' in row and isinstance(row['keywords'], list):
                    st.write("**Keywords:**")
                    keywords_html = " ".join([f"<span style='background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 10px;'>{keyword}</span>" for keyword in row['keywords']])
                    st.markdown(keywords_html, unsafe_allow_html=True)
                
                # Display relevant departments
                if 'relevant_departments' in row and isinstance(row['relevant_departments'], list):
                    st.write("**Relevant Departments:**")
                    st.write(", ".join(row['relevant_departments']))
                
                # Link to grant details
                if 'url' in row and row['url']:
                    st.markdown(f"[View Grant Details]({row['url']})")
    else:
        st.warning("No grants match your search criteria. Try adjusting your filters.")

# =======================================================
# TAB 2: ANALYTICS
# =======================================================
with tab2:
    st.header("Grant Analytics Dashboard")
    
    # Create two columns for analytics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Funding by Agency")
        # Group data by funding agency
        agency_funding = grants_df.groupby('funding_agency')['award_amount'].sum().sort_values(ascending=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        agency_funding.plot(kind='barh', ax=ax)
        ax.set_xlabel('Total Funding Amount ($)')
        ax.set_ylabel('Funding Agency')
        ax.set_title('Total Funding by Agency')
        # Format x-axis to show in millions
        ax.xaxis.set_major_formatter(lambda x, pos: f'${x/1e6:.1f}M')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Grant Deadlines")
        # Convert deadline to datetime
        grants_df['deadline_dt'] = pd.to_datetime(grants_df['deadline'])
        
        # Group by month
        grants_df['deadline_month'] = grants_df['deadline_dt'].dt.strftime('%Y-%m')
        monthly_counts = grants_df.groupby('deadline_month').size()
        
        # Create line chart
        fig, ax = plt.subplots(figsize=(10, 6))
        monthly_counts.plot(kind='line', marker='o', ax=ax)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Grants')
        ax.set_title('Grant Deadlines by Month')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Full width analytics
    st.subheader("Grant Amount Distribution")
    
    # Create histogram of award amounts
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 3000001, 250000)
    ax.hist(grants_df['award_amount'], bins=bins, color='skyblue', edgecolor='black')
    ax.set_xlabel('Award Amount ($)')
    ax.set_ylabel('Number of Grants')
    ax.set_title('Distribution of Grant Award Amounts')
    # Format x-axis to show in millions
    ax.xaxis.set_major_formatter(lambda x, pos: f'${x/1e6:.1f}M')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Department focus analysis
    st.subheader("Funding by Department")
    
    # Count occurrences of each department
    dept_counts = {}
    dept_funding = {}
    
    for _, row in grants_df.iterrows():
        if isinstance(row['relevant_departments'], list):
            for dept in row['relevant_departments']:
                if dept in dept_counts:
                    dept_counts[dept] += 1
                    dept_funding[dept] += row['award_amount']
                else:
                    dept_counts[dept] = 1
                    dept_funding[dept] = row['award_amount']
    
    # Convert to DataFrame for plotting
    dept_df = pd.DataFrame({
        'Department': list(dept_counts.keys()),
        'Grant Count': list(dept_counts.values()),
        'Total Funding': list(dept_funding.values())
    }).sort_values('Total Funding', ascending=False)
    
    # Create two columns
    col3, col4 = st.columns(2)
    
    with col3:
        # Create bar chart of department counts
        fig, ax = plt.subplots(figsize=(10, 8))
        dept_df.set_index('Department')['Grant Count'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Number of Grants')
        ax.set_ylabel('Department')
        ax.set_title('Grant Opportunities by Department')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col4:
        # Create bar chart of department funding
        fig, ax = plt.subplots(figsize=(10, 8))
        dept_df.set_index('Department')['Total Funding'].plot(kind='barh', ax=ax)
        ax.set_xlabel('Total Funding Amount ($)')
        ax.set_ylabel('Department')
        ax.set_title('Total Funding Available by Department')
        # Format x-axis to show in millions
        ax.xaxis.set_major_formatter(lambda x, pos: f'${x/1e6:.1f}M')
        plt.tight_layout()
        st.pyplot(fig)

# =======================================================
# TAB 3: INSTITUTIONAL PROFILE
# =======================================================
with tab3:
    # Call the institutional profile function
    add_institutional_profile_to_app()

# =======================================================
# TAB 4: ABOUT
# =======================================================
with tab4:
    st.header("About Healthcare Grants Explorer")
    
    st.write("""
    ## Overview
    
    The Healthcare Grants Explorer is a specialized tool designed to help Sheba Medical Center identify, 
    track, and apply for healthcare research grants that align with the institution's research priorities 
    and departmental strengths.
    
    ## Features
    
    - **Grant Explorer**: Search and filter grant opportunities by keyword, funding agency, department, and award amount
    - **Analytics Dashboard**: Visualize grant data to identify trends and opportunities
    - **Institutional Profile**: View Sheba Medical Center's research focus areas, departments, key researchers, and match grants to institutional strengths
    
    ## Development
    
    This application is developed using Python and the Streamlit framework, making it easy to use and accessible 
    through any web browser. The application is designed to be extendable with new features and integrations.
    
    ## Future Enhancements
    
    - Integration with external grant databases (NIH, EU Horizon, etc.)
    - Email notifications for new matching grants
    - Grant application tracking and collaboration tools
    - Machine learning for better grant matching
    
    ## Contact
    
    For questions, feedback, or feature requests, please contact the development team at [example@email.com].
    """)
    
    # Credits section with columns for team members
    st.subheader("Development Team")
    
    team_cols = st.columns(3)
    
    with team_cols[0]:
        st.image("https://via.placeholder.com/150", width=150)
        st.write("**Dr. Aurel Shelby**")
        st.write("Project Lead")
    
    with team_cols[1]:
        st.image("https://via.placeholder.com/150", width=150)
        st.write("**Dr. Sarah Cohen**")
        st.write("Scientific Advisor")
    
    with team_cols[2]:
        st.image("https://via.placeholder.com/150", width=150)
        st.write("**Michael Davis**")
        st.write("Data Scientist")
    
    # Version information
    st.markdown("---")
    st.write("Version 1.0.0 | Last Updated: March 2025")
    st.write("© 2025 Sheba Medical Center. All rights reserved.")

# Add a footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Healthcare Grants Explorer | Sheba Medical Center</div>", unsafe_allow_html=True)