import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

# ========== DATA MODELS ==========

class ResearchFocusArea:
    def __init__(
        self,
        name: str,
        description: str,
        strength_rating: int,  # 1-10 scale
        key_accomplishments: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.strength_rating = strength_rating
        self.key_accomplishments = key_accomplishments or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "strength_rating": self.strength_rating,
            "key_accomplishments": self.key_accomplishments
        }


class Department:
    def __init__(
        self,
        name: str,
        description: str,
        head_name: str,
        specialties: List[str],
        staff_count: Optional[int] = None,
        equipment_highlights: Optional[List[str]] = None
    ):
        self.name = name
        self.description = description
        self.head_name = head_name
        self.specialties = specialties
        self.staff_count = staff_count
        self.equipment_highlights = equipment_highlights or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "head_name": self.head_name,
            "specialties": self.specialties,
            "staff_count": self.staff_count,
            "equipment_highlights": self.equipment_highlights
        }


class Researcher:
    def __init__(
        self,
        name: str,
        title: str,
        department: str,
        specialties: List[str],
        publications: Optional[int] = None,
        h_index: Optional[int] = None,
        profile_url: Optional[str] = None,
        photo_url: Optional[str] = None,
        email: Optional[str] = None
    ):
        self.name = name
        self.title = title
        self.department = department
        self.specialties = specialties
        self.publications = publications
        self.h_index = h_index
        self.profile_url = profile_url
        self.photo_url = photo_url
        self.email = email

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "department": self.department,
            "specialties": self.specialties,
            "publications": self.publications,
            "h_index": self.h_index,
            "profile_url": self.profile_url,
            "photo_url": self.photo_url,
            "email": self.email
        }


class PastGrant:
    def __init__(
        self,
        title: str,
        funding_agency: str,
        amount: float,
        year: int,
        principal_investigator: str,
        department: str,
        status: str,  # "Completed", "Ongoing", or "Awarded"
        outcomes: Optional[List[str]] = None
    ):
        self.title = title
        self.funding_agency = funding_agency
        self.amount = amount
        self.year = year
        self.principal_investigator = principal_investigator
        self.department = department
        self.status = status
        self.outcomes = outcomes or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "funding_agency": self.funding_agency,
            "amount": self.amount,
            "year": self.year,
            "principal_investigator": self.principal_investigator,
            "department": self.department,
            "status": self.status,
            "outcomes": self.outcomes
        }


class Infrastructure:
    def __init__(
        self,
        name: str,
        type: str,
        description: str,
        year_acquired: Optional[int] = None,
        highlight_features: Optional[List[str]] = None
    ):
        self.name = name
        self.type = type
        self.description = description
        self.year_acquired = year_acquired
        self.highlight_features = highlight_features or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "year_acquired": self.year_acquired,
            "highlight_features": self.highlight_features
        }


class Institution:
    def __init__(
        self,
        name: str,
        location: Dict[str, Any],
        overview: str,
        founded_year: int,
        institution_type: str,
        size: Dict[str, Any],
        contact: Dict[str, Any],
        research_focus_areas: List[ResearchFocusArea],
        departments: List[Department],
        researchers: List[Researcher],
        past_grants: List[PastGrant],
        infrastructure: List[Infrastructure],
        collaborations: Optional[List[Dict[str, Any]]] = None,
        publications: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.location = location
        self.overview = overview
        self.founded_year = founded_year
        self.institution_type = institution_type
        self.size = size
        self.contact = contact
        self.research_focus_areas = research_focus_areas
        self.departments = departments
        self.researchers = researchers
        self.past_grants = past_grants
        self.infrastructure = infrastructure
        self.collaborations = collaborations or []
        self.publications = publications or {
            "total": 0,
            "by_year": [],
            "top_journals": [],
            "highlight_publications": []
        }
        self.metrics = metrics or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "location": self.location,
            "overview": self.overview,
            "founded_year": self.founded_year,
            "institution_type": self.institution_type,
            "size": self.size,
            "contact": self.contact,
            "research_focus_areas": [area.to_dict() for area in self.research_focus_areas],
            "departments": [dept.to_dict() for dept in self.departments],
            "researchers": [researcher.to_dict() for researcher in self.researchers],
            "past_grants": [grant.to_dict() for grant in self.past_grants],
            "infrastructure": [item.to_dict() for item in self.infrastructure],
            "collaborations": self.collaborations,
            "publications": self.publications,
            "metrics": self.metrics
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Institution':
        research_focus_areas = [
            ResearchFocusArea(**area_data) 
            for area_data in data.get("research_focus_areas", [])
        ]
        
        departments = [
            Department(**dept_data) 
            for dept_data in data.get("departments", [])
        ]
        
        researchers = [
            Researcher(**researcher_data) 
            for researcher_data in data.get("researchers", [])
        ]
        
        past_grants = [
            PastGrant(**grant_data) 
            for grant_data in data.get("past_grants", [])
        ]
        
        infrastructure = [
            Infrastructure(**infra_data) 
            for infra_data in data.get("infrastructure", [])
        ]
        
        return cls(
            name=data["name"],
            location=data["location"],
            overview=data["overview"],
            founded_year=data["founded_year"],
            institution_type=data["institution_type"],
            size=data["size"],
            contact=data["contact"],
            research_focus_areas=research_focus_areas,
            departments=departments,
            researchers=researchers,
            past_grants=past_grants,
            infrastructure=infrastructure,
            collaborations=data.get("collaborations", []),
            publications=data.get("publications", {}),
            metrics=data.get("metrics", {})
        )

    def save_to_json(self, filepath: str) -> None:
        """Save institution data to a JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            st.warning(f"Could not save institution data to {filepath}: {str(e)}")

    @classmethod
    def load_from_json(cls, filepath: str) -> 'Institution':
        """Load institution data from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ========== GRANT MATCHING ALGORITHM ==========

def calculate_grant_match(
    institution: Institution, 
    grant: Dict[str, Any],
    keywords_weight: float = 0.6,
    departments_weight: float = 0.3,
    funding_agency_weight: float = 0.1
) -> float:
    """
    Calculate a match score between an institution and a grant opportunity.
    
    Args:
        institution: The institution to match against
        grant: The grant opportunity to evaluate
        keywords_weight: Weight for keyword matching
        departments_weight: Weight for department relevance
        funding_agency_weight: Weight for funding agency history
        
    Returns:
        Float between 0 and 100 representing match percentage
    """
    # Extract keywords from the grant
    grant_keywords = set()
    for field in ['title', 'description', 'keywords']:
        if field in grant and grant[field]:
            if isinstance(grant[field], list):
                grant_keywords.update([k.lower() for k in grant[field]])
            else:
                # Split text on spaces, commas, etc. and add to keywords
                words = str(grant[field]).lower().split()
                grant_keywords.update(words)
    
    # Calculate keyword match with research focus areas
    keyword_score = 0
    for area in institution.research_focus_areas:
        area_keywords = set(area.name.lower().split() + area.description.lower().split())
        
        # Count matches
        matches = len(grant_keywords.intersection(area_keywords))
        if matches > 0:
            # Weight by strength rating and normalize
            keyword_score += (matches / len(grant_keywords)) * (area.strength_rating / 10)
    
    # Normalize keyword score
    if institution.research_focus_areas:
        keyword_score /= len(institution.research_focus_areas)
    
    # Calculate department relevance
    department_score = 0
    if 'relevant_departments' in grant and grant['relevant_departments']:
        relevant_depts = [d.lower() for d in grant['relevant_departments']]
        for dept in institution.departments:
            dept_name_lower = dept.name.lower()
            if dept_name_lower in relevant_depts:
                department_score += 1
                continue
                
            # Check if any specialties match
            for specialty in dept.specialties:
                if any(specialty.lower() in kw for kw in grant_keywords):
                    department_score += 0.5
                    break
    
    # Normalize department score
    if institution.departments:
        department_score /= len(institution.departments)
    
    # Calculate funding agency history
    agency_score = 0
    if 'funding_agency' in grant and grant['funding_agency']:
        # Check past grants from the same agency
        agency_name = grant['funding_agency'].lower()
        past_from_agency = [g for g in institution.past_grants 
                          if g.funding_agency.lower() == agency_name]
        
        if past_from_agency:
            # Higher score if there's history with this agency
            agency_score = min(1.0, len(past_from_agency) / 5)  # Cap at 1.0
    
    # Calculate final weighted score
    final_score = (
        keyword_score * keywords_weight + 
        department_score * departments_weight + 
        agency_score * funding_agency_weight
    ) * 100
    
    # Cap at 100
    return min(final_score, 100)


def get_matching_grants(
    institution: Institution,
    grants_df: pd.DataFrame,
    threshold: float = 30.0
) -> pd.DataFrame:
    """
    Find grants that match the institution's profile.
    
    Args:
        institution: The institution to match against
        grants_df: DataFrame containing grant opportunities
        threshold: Minimum match score to include (0-100)
        
    Returns:
        DataFrame of grants with match scores, sorted by score
    """
    # Convert DataFrame rows to dictionaries
    grants = grants_df.to_dict('records')
    
    # Calculate match scores for each grant
    grants_with_scores = []
    for grant in grants:
        match_score = calculate_grant_match(institution, grant)
        if match_score >= threshold:
            grant_copy = grant.copy()
            grant_copy['match_score'] = match_score
            grants_with_scores.append(grant_copy)
    
    # Convert back to DataFrame and sort by match score
    results_df = pd.DataFrame(grants_with_scores)
    if not results_df.empty:
        results_df = results_df.sort_values('match_score', ascending=False)
    
    return results_df


# ========== STREAMLIT UI COMPONENTS ==========

def display_institution_header(institution: Institution):
    """Display the institution header with basic information"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title(institution.name)
        st.subheader(f"{institution.location['city']}, {institution.location['country']}")
        st.write(institution.overview)
    
    with col2:
        # Placeholder for institution logo or image
        st.image("https://via.placeholder.com/150", width=150)
        
        # Contact links
        st.write(f"**Website:** [{institution.contact['website']}]({institution.contact['website']})")
        st.write(f"**Email:** {institution.contact['email']}")
        if 'phone' in institution.contact and institution.contact['phone']:
            st.write(f"**Phone:** {institution.contact['phone']}")


def display_institution_facts(institution: Institution):
    """Display key facts about the institution"""
    st.subheader("Institution Facts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Type:** {institution.institution_type}")
        st.write(f"**Founded:** {institution.founded_year}")
        
        if 'beds' in institution.size and institution.size['beds']:
            st.write(f"**Beds:** {institution.size['beds']:,}")
            
        if 'annual_patients' in institution.size and institution.size['annual_patients']:
            st.write(f"**Annual Patients:** {institution.size['annual_patients']:,}")
    
    with col2:
        if 'research_budget' in institution.size and institution.size['research_budget']:
            st.write(f"**Research Budget:** ${institution.size['research_budget']:,}")
            
        if 'grant_success_rate' in institution.size and institution.size['grant_success_rate']:
            st.write(f"**Grant Success Rate:** {institution.size['grant_success_rate']}%")
            
        if 'employees' in institution.size and institution.size['employees']:
            st.write(f"**Employees:** {institution.size['employees']:,}")


def display_research_focus_areas(institution: Institution):
    """Display research focus areas with strength ratings"""
    st.subheader("Research Focus Areas")
    
    for area in institution.research_focus_areas:
        with st.expander(f"{area.name} - Strength: {area.strength_rating}/10"):
            st.write(area.description)
            
            if area.key_accomplishments:
                st.write("**Key Accomplishments:**")
                for accomplishment in area.key_accomplishments:
                    st.write(f"- {accomplishment}")
                    
        # Also show a progress bar for strength
        col1, col2 = st.columns([3, 1])
        with col1:
            st.progress(area.strength_rating / 10)
        with col2:
            st.write(f"{area.strength_rating}/10")


def display_departments(institution: Institution):
    """Display departments with their specialties"""
    st.subheader("Departments")
    
    # Create 2 columns for departments
    cols = st.columns(2)
    
    # Alternate departments between columns
    for i, dept in enumerate(institution.departments):
        with cols[i % 2]:
            with st.expander(dept.name):
                st.write(f"**Head:** {dept.head_name}")
                st.write(dept.description)
                
                st.write("**Specialties:**")
                for specialty in dept.specialties:
                    st.markdown(f"- {specialty}")
                
                if dept.equipment_highlights:
                    st.write("**Key Equipment:**")
                    for equipment in dept.equipment_highlights:
                        st.markdown(f"- {equipment}")
                        
                if dept.staff_count:
                    st.write(f"**Staff Count:** {dept.staff_count}")


def display_researchers(institution: Institution, search_term: str = None):
    """Display researchers with filtering capabilities"""
    st.subheader("Key Researchers")
    
    # Search box
    search = st.text_input("Search researchers by name or specialty", 
                           value=search_term if search_term else "")
    
    # Filter researchers based on search
    filtered_researchers = institution.researchers
    if search:
        search_lower = search.lower()
        filtered_researchers = [
            r for r in institution.researchers
            if (search_lower in r.name.lower() or 
                search_lower in r.department.lower() or
                any(search_lower in s.lower() for s in r.specialties))
        ]
    
    # Display count of matches
    st.write(f"Showing {len(filtered_researchers)} of {len(institution.researchers)} researchers")
    
    # Create 3 columns for researchers
    cols = st.columns(3)
    
    # Alternate researchers between columns
    for i, researcher in enumerate(filtered_researchers):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"### {researcher.name}")
                st.write(f"**Title:** {researcher.title}")
                st.write(f"**Department:** {researcher.department}")
                
                st.write("**Specialties:**")
                specialty_str = ", ".join(researcher.specialties)
                st.write(specialty_str)
                
                metrics_cols = st.columns(2)
                with metrics_cols[0]:
                    if researcher.publications:
                        st.metric("Publications", researcher.publications)
                with metrics_cols[1]:
                    if researcher.h_index:
                        st.metric("h-index", researcher.h_index)
                
                if researcher.profile_url:
                    st.markdown(f"[View Profile]({researcher.profile_url})")
                
                st.markdown("---")


def display_grants_history(institution: Institution, department_filter: str = None):
    """Display grant history with filtering capabilities"""
    st.subheader("Grant History")
    
    # Filter by department
    departments = ["All Departments"] + [d.name for d in institution.departments]
    selected_dept = st.selectbox("Filter by Department", departments, 
                                index=departments.index(department_filter) if department_filter in departments else 0)
    
    # Filter grants based on department selection
    filtered_grants = institution.past_grants
    if selected_dept != "All Departments":
        filtered_grants = [g for g in institution.past_grants if g.department == selected_dept]
    
    # Convert to DataFrame for easier display
    if filtered_grants:
        grants_data = [g.to_dict() for g in filtered_grants]
        grants_df = pd.DataFrame(grants_data)
        
        # Format columns
        grants_df['amount'] = grants_df['amount'].apply(lambda x: f"${x:,.2f}")
        
        # Sort by year (most recent first)
        grants_df = grants_df.sort_values('year', ascending=False)
        
        # Display as a table
        st.dataframe(
            grants_df[['title', 'funding_agency', 'year', 'principal_investigator', 
                       'department', 'amount', 'status']],
            use_container_width=True
        )
        
        # Calculate total funding
        total_funding = sum(g.amount for g in filtered_grants)
        st.write(f"**Total Funding:** ${total_funding:,.2f}")
        
        # Display funding by year chart using matplotlib
        funding_by_year = {}
        for grant in filtered_grants:
            if grant.year in funding_by_year:
                funding_by_year[grant.year] += grant.amount
            else:
                funding_by_year[grant.year] = grant.amount
        
        funding_df = pd.DataFrame({
            'Year': list(funding_by_year.keys()),
            'Amount': list(funding_by_year.values())
        })
        funding_df = funding_df.sort_values('Year')
        
        st.subheader('Funding by Year')
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(funding_df['Year'].astype(str), funding_df['Amount'])
        
        # Format y-axis to show in millions
        ax.yaxis.set_major_formatter(lambda x, pos: f'${x/1e6:.1f}M')
        
        # Add labels and title
        ax.set_xlabel('Year')
        ax.set_ylabel('Funding Amount')
        ax.set_title('Funding by Year')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid for better readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Tight layout to ensure everything fits
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
    else:
        st.write("No grants found matching the selected filters.")


def display_matching_grants(institution: Institution, grants_df: pd.DataFrame, department_filter: str = None):
    """Display grants that match the institution profile"""
    st.subheader("Matching Grant Opportunities")
    
    # Filter by department if needed
    departments = ["All Departments"] + [d.name for d in institution.departments]
    selected_dept = st.selectbox("Filter by Department", departments, 
                                index=departments.index(department_filter) if department_filter in departments else 0,
                                key="matching_grants_dept")
    
    # Get matching grants
    matching_grants = get_matching_grants(institution, grants_df)
    
    if matching_grants.empty:
        st.write("No matching grants found.")
        return
    
    # Filter by department if selected
    if selected_dept != "All Departments" and 'relevant_departments' in matching_grants.columns:
        # This assumes relevant_departments is a list column
        matching_grants = matching_grants[
            matching_grants['relevant_departments'].apply(
                lambda x: selected_dept in x if isinstance(x, list) else False
            )
        ]
    
    # Format the DataFrame for display
    display_df = matching_grants.copy()
    
    # Format columns
    if 'award_amount' in display_df.columns:
        display_df['award_amount'] = display_df['award_amount'].apply(
            lambda x: f"${x:,.2f}" if pd.notnull(x) else "Varies"
        )
    
    if 'deadline' in display_df.columns:
        display_df['deadline'] = pd.to_datetime(display_df['deadline']).dt.strftime('%Y-%m-%d')
    
    # Round match score to 1 decimal place
    display_df['match_score'] = display_df['match_score'].round(1)
    
    # Select columns to display
    display_columns = ['title', 'funding_agency', 'deadline', 'award_amount', 'match_score']
    display_columns = [col for col in display_columns if col in display_df.columns]
    
    # Add match score visualization
    st.dataframe(
        display_df[display_columns],
        use_container_width=True
    )
    
    # Visualize match scores using matplotlib
    top_matches = matching_grants.head(10).sort_values('match_score')  # Sort for better visualization
    
    st.subheader('Top 10 Grant Matches')
    
    # Create matplotlib figure for horizontal bar chart of match scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Limit title length for better display
    short_titles = [title[:40] + "..." if len(title) > 40 else title 
                   for title in top_matches['title']]
    
    # Create horizontal bar chart
    bars = ax.barh(short_titles, top_matches['match_score'])
    
    # Color the bars based on match score
    colors = ['#d4e6f1' if score < 50 else 
              '#a9cce3' if score < 70 else 
              '#5499c7' if score < 85 else 
              '#2e86c1' for score in top_matches['match_score']]
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    # Add score values at the end of each bar
    for i, (score, title) in enumerate(zip(top_matches['match_score'], short_titles)):
        ax.text(score + 1, i, f"{score:.1f}%", va='center')
    
    # Set chart limits and labels
    ax.set_xlim(0, 105)  # Give some space for the text
    ax.set_xlabel('Match Score (%)')
    ax.set_title('Grant Match Scores')
    
    # Remove y-axis label as the bar labels are self-explanatory
    ax.set_ylabel('')
    
    # Add grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout to ensure everything fits
    plt.tight_layout()
    
    # Display the chart
    st.pyplot(fig)


def institutional_profile_tab(institution: Institution, grants_df: pd.DataFrame):
    """Main function to display the institutional profile tab"""
    # Create tabs for different sections
    tabs = st.tabs([
        "Overview", 
        "Departments", 
        "Researchers", 
        "Grant History",
        "Matching Grants",
        "Infrastructure"
    ])
    
    # Overview tab
    with tabs[0]:
        display_institution_header(institution)
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            display_institution_facts(institution)
        with col2:
            # Display a map if we have coordinates
            if ('coordinates' in institution.location and 
                institution.location['coordinates'] and
                'latitude' in institution.location['coordinates'] and
                'longitude' in institution.location['coordinates']):
                
                lat = institution.location['coordinates']['latitude']
                lon = institution.location['coordinates']['longitude']
                
                location_df = pd.DataFrame({
                    'lat': [lat],
                    'lon': [lon]
                })
                
                st.map(location_df)
        
        st.markdown("---")
        display_research_focus_areas(institution)
    
    # Departments tab
    with tabs[1]:
        display_departments(institution)
    
    # Researchers tab
    with tabs[2]:
        display_researchers(institution)
    
    # Grant History tab
    with tabs[3]:
        display_grants_history(institution)
    
    # Matching Grants tab
    with tabs[4]:
        display_matching_grants(institution, grants_df)
    
    # Infrastructure tab
    with tabs[5]:
        st.subheader("Research Infrastructure")
        
        # Group infrastructure by type
        infra_by_type = {}
        for item in institution.infrastructure:
            if item.type not in infra_by_type:
                infra_by_type[item.type] = []
            infra_by_type[item.type].append(item)
        
        # Display by type
        for infra_type, items in infra_by_type.items():
            st.write(f"### {infra_type}")
            
            for item in items:
                with st.expander(item.name):
                    st.write(item.description)
                    
                    if item.highlight_features:
                        st.write("**Key Features:**")
                        for feature in item.highlight_features:
                            st.markdown(f"- {feature}")
                    
                    if item.year_acquired:
                        st.write(f"**Year Acquired:** {item.year_acquired}")


# ========== SAMPLE DATA ==========

def create_sheba_medical_center() -> Institution:
    """Create sample data for Sheba Medical Center"""
    research_focus_areas = [
        ResearchFocusArea(
            name="Oncology",
            description="Advanced cancer research focusing on precision medicine and immunotherapy",
            strength_rating=9,
            key_accomplishments=[
                "Development of CAR-T cell therapies for solid tumors",
                "Pioneering work in cancer genomics and personalized treatment",
                "Clinical trials for novel cancer immunotherapies"
            ]
        ),
        ResearchFocusArea(
            name="Cardiology",
            description="Innovative cardiovascular research and advanced cardiac procedures",
            strength_rating=8,
            key_accomplishments=[
                "Development of minimally invasive cardiac procedures",
                "Novel treatments for heart failure",
                "Advanced imaging techniques for cardiac diagnosis"
            ]
        ),
        ResearchFocusArea(
            name="Neuroscience",
            description="Cutting-edge research in neurodegenerative diseases and brain disorders",
            strength_rating=7,
            key_accomplishments=[
                "Novel therapeutic approaches for Alzheimer's disease",
                "Advanced neuroimaging techniques",
                "Neural stem cell research for regenerative medicine"
            ]
        ),
        ResearchFocusArea(
            name="Pediatrics",
            description="Comprehensive research in pediatric diseases and treatments",
            strength_rating=9,
            key_accomplishments=[
                "Pioneering treatments for rare pediatric disorders",
                "Development of pediatric surgical techniques",
                "Novel diagnostic approaches for childhood diseases"
            ]
        ),
        ResearchFocusArea(
            name="Rehabilitation Medicine",
            description="Innovative approaches to physical rehabilitation and recovery",
            strength_rating=10,
            key_accomplishments=[
                "Virtual reality applications in rehabilitation",
                "Robotic-assisted rehabilitation techniques",
                "Novel therapies for spinal cord injuries"
            ]
        )
    ]
    
    departments = [
        Department(
            name="Department of Oncology",
            description="Comprehensive cancer care and research department focusing on novel therapies",
            head_name="Dr. Sarah Cohen",
            specialties=["Medical Oncology", "Radiation Oncology", "Surgical Oncology", "Cancer Genomics"],
            staff_count=135,
            equipment_highlights=[
                "Linear Accelerators for precise radiation therapy",
                "Next-generation sequencing lab for cancer genomics",
                "Advanced imaging systems for cancer diagnosis"
            ]
        ),
        Department(
            name="Heart Institute",
            description="Comprehensive cardiac care center with cutting-edge research facilities",
            head_name="Dr. Daniel Levy",
            specialties=["Interventional Cardiology", "Cardiac Surgery", "Electrophysiology", "Heart Failure"],
            staff_count=120,
            equipment_highlights=[
                "Cardiac Catheterization Labs",
                "3D Echocardiography",
                "Advanced Cardiac MRI"
            ]
        ),
        Department(
            name="Neuroscience Center",
            description="Multidisciplinary center for neurological disorders and research",
            head_name="Dr. Rachel Stern",
            specialties=["Neurology", "Neurosurgery", "Neuroimaging", "Neuroimmunology"],
            staff_count=85,
            equipment_highlights=[
                "3T MRI Scanner for neuroimaging",
                "Neurophysiology lab for EEG and EMG studies",
                "Stereotactic navigation system for neurosurgery"
            ]
        ),
        Department(
            name="Pediatric Medical Center",
            description="Comprehensive care for children with complex medical conditions",
            head_name="Dr. Michael Rosen",
            specialties=["Pediatric Oncology", "Pediatric Cardiology", "Neonatology", "Pediatric Surgery"],
            staff_count=160,
            equipment_highlights=[
                "Neonatal Intensive Care Units",
                "Pediatric advanced life support systems",
                "Child-friendly diagnostic equipment"
            ]
        ),
        Department(
            name="Rehabilitation Center",
            description="Israel's leading rehabilitation center with innovative technology",
            head_name="Dr. Liora Goldstein",
            specialties=["Physical Therapy", "Occupational Therapy", "Speech Therapy", "Neurological Rehabilitation"],
            staff_count=110,
            equipment_highlights=[
                "Robotic exoskeletons for gait training",
                "Virtual reality rehabilitation systems",
                "Advanced prosthetics and orthotics lab"
            ]
        )
    ]
    
    researchers = [
        Researcher(
            name="Dr. Sarah Cohen",
            title="Director of Oncology",
            department="Department of Oncology",
            specialties=["Cancer Immunotherapy", "Precision Oncology", "Clinical Trials"],
            publications=156,
            h_index=42,
            email="sarah.cohen@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Daniel Levy",
            title="Chief of Cardiology",
            department="Heart Institute",
            specialties=["Interventional Cardiology", "Heart Failure", "Cardiac Imaging"],
            publications=132,
            h_index=38,
            email="daniel.levy@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Rachel Stern",
            title="Head of Neuroscience",
            department="Neuroscience Center",
            specialties=["Neurodegeneration", "Stroke Research", "Neuroimmunology"],
            publications=115,
            h_index=32,
            email="rachel.stern@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Michael Rosen",
            title="Director of Pediatrics",
            department="Pediatric Medical Center",
            specialties=["Pediatric Oncology", "Rare Diseases", "Genetic Disorders"],
            publications=118,
            h_index=35,
            email="michael.rosen@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Liora Goldstein",
            title="Head of Rehabilitation",
            department="Rehabilitation Center",
            specialties=["Neurological Rehabilitation", "Spinal Cord Injury", "Robotic Rehabilitation"],
            publications=93,
            h_index=28,
            email="liora.goldstein@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. David Katz",
            title="Senior Researcher",
            department="Department of Oncology",
            specialties=["Cancer Genomics", "Bioinformatics", "Targeted Therapy"],
            publications=84,
            h_index=26,
            email="david.katz@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Aviva Berger",
            title="Principal Investigator",
            department="Neuroscience Center",
            specialties=["Multiple Sclerosis", "Neuroimmunology", "Clinical Trials"],
            publications=76,
            h_index=24,
            email="aviva.berger@sheba.health.gov.il"
        ),
        Researcher(
            name="Dr. Samuel Gold",
            title="Head of Research",
            department="Heart Institute",
            specialties=["Cardiac Electrophysiology", "Arrhythmias", "Cardiac Devices"],
            publications=92,
            h_index=30,
            email="samuel.gold@sheba.health.gov.il"
        )
    ]
    
    past_grants = [
        PastGrant(
            title="Targeting Cancer Stem Cells in Resistant Tumors",
            funding_agency="National Institutes of Health",
            amount=2750000,
            year=2021,
            principal_investigator="Dr. Sarah Cohen",
            department="Department of Oncology",
            status="Ongoing"
        ),
        PastGrant(
            title="Novel Immunotherapy Approaches for Solid Tumors",
            funding_agency="Cancer Research Foundation",
            amount=1850000,
            year=2020,
            principal_investigator="Dr. David Katz",
            department="Department of Oncology",
            status="Ongoing"
        ),
        PastGrant(
            title="Cardiac Regeneration Using Stem Cell Therapy",
            funding_agency="American Heart Association",
            amount=1250000,
            year=2019,
            principal_investigator="Dr. Daniel Levy",
            department="Heart Institute",
            status="Completed",
            outcomes=[
                "Development of novel cardiac stem cell isolation technique",
                "Successful Phase I clinical trial with 24 patients",
                "Publication in major cardiology journals"
            ]
        ),
        PastGrant(
            title="Neuroinflammation in Alzheimer's Disease",
            funding_agency="Alzheimer's Research Foundation",
            amount=980000,
            year=2020,
            principal_investigator="Dr. Rachel Stern",
            department="Neuroscience Center",
            status="Ongoing"
        ),
        PastGrant(
            title="Robotic Systems for Neurological Rehabilitation",
            funding_agency="National Science Foundation",
            amount=1450000,
            year=2018,
            principal_investigator="Dr. Liora Goldstein",
            department="Rehabilitation Center",
            status="Completed",
            outcomes=[
                "Development of novel robotic exoskeleton",
                "Clinical validation with 45 stroke patients",
                "Two patents filed for rehabilitation technology"
            ]
        ),
        PastGrant(
            title="Genetic Markers for Pediatric Cancer Risk",
            funding_agency="Pediatric Cancer Foundation",
            amount=1350000,
            year=2019,
            principal_investigator="Dr. Michael Rosen",
            department="Pediatric Medical Center",
            status="Completed",
            outcomes=[
                "Identification of three novel genetic markers",
                "Development of screening protocol",
                "Publication in major pediatric oncology journals"
            ]
        ),
        PastGrant(
            title="Cardiac Device Optimization for Heart Failure",
            funding_agency="Medical Research Council",
            amount=950000,
            year=2021,
            principal_investigator="Dr. Samuel Gold",
            department="Heart Institute",
            status="Ongoing"
        ),
        PastGrant(
            title="Multiple Sclerosis Biomarkers for Treatment Response",
            funding_agency="National Institutes of Health",
            amount=1650000,
            year=2020,
            principal_investigator="Dr. Aviva Berger",
            department="Neuroscience Center",
            status="Ongoing"
        )
    ]
    
    infrastructure = [
        Infrastructure(
            name="Cancer Research Laboratory",
            type="Research Lab",
            description="State-of-the-art laboratory for cancer research with genomics and proteomics capabilities",
            year_acquired=2018,
            highlight_features=[
                "Next-generation sequencing facility",
                "Mass spectrometry for proteomics",
                "Cell culture and animal model facilities",
                "Bioinformatics computing cluster"
            ]
        ),
        Infrastructure(
            name="Advanced Imaging Center",
            type="Diagnostic Facility",
            description="Comprehensive imaging center with the latest technology for clinical and research use",
            year_acquired=2019,
            highlight_features=[
                "3T MRI Scanner",
                "PET-CT Scanner",
                "Digital Radiography Systems",
                "Advanced Ultrasound Systems"
            ]
        ),
        Infrastructure(
            name="Clinical Trials Unit",
            type="Clinical Research",
            description="Dedicated facility for conducting clinical trials with full regulatory compliance",
            year_acquired=2017,
            highlight_features=[
                "20 dedicated beds for research patients",
                "Specialized pharmacy for investigational drugs",
                "Dedicated staff for trial management",
                "Data management and biostatistics support"
            ]
        ),
        Infrastructure(
            name="Rehabilitation Technology Center",
            type="Therapeutic Facility",
            description="Advanced center for rehabilitation technology and research",
            year_acquired=2020,
            highlight_features=[
                "Virtual reality rehabilitation systems",
                "Robotic gait training equipment",
                "Computer-assisted therapy programs",
                "Motion analysis laboratory"
            ]
        ),
        Infrastructure(
            name="Cardiac Catheterization Lab",
            type="Clinical Facility",
            description="Advanced lab for cardiac procedures and research",
            year_acquired=2019,
            highlight_features=[
                "Biplane angiography system",
                "3D cardiac mapping system",
                "Intravascular ultrasound",
                "Fractional flow reserve measurement"
            ]
        )
    ]
    
    publications = {
        "total": 12850,
        "by_year": [
            {"year": 2017, "count": 2100, "fundingAmount": 9500000},
            {"year": 2018, "count": 2250, "fundingAmount": 10200000},
            {"year": 2019, "count": 2450, "fundingAmount": 12100000},
            {"year": 2020, "count": 2750, "fundingAmount": 13500000},
            {"year": 2021, "count": 3300, "fundingAmount": 14800000}
        ],
        "top_journals": [
            {"journal": "New England Journal of Medicine", "count": 45},
            {"journal": "The Lancet", "count": 38},
            {"journal": "JAMA", "count": 32},
            {"journal": "Nature Medicine", "count": 29},
            {"journal": "Science Translational Medicine", "count": 25}
        ],
        "highlight_publications": [
            {
                "title": "Novel CAR-T Cell Therapy for Resistant Solid Tumors",
                "journal": "New England Journal of Medicine",
                "year": 2021,
                "doi": "10.1056/NEJMoa2035451",
                "citations": 156
            },
            {
                "title": "Cardiac Regeneration Using Modified Stem Cells",
                "journal": "Nature Medicine",
                "year": 2020,
                "doi": "10.1038/s41591-020-0845-1",
                "citations": 112
            },
            {
                "title": "Virtual Reality for Stroke Rehabilitation",
                "journal": "The Lancet Neurology",
                "year": 2019,
                "doi": "10.1016/S1474-4422(19)30034-8",
                "citations": 98
            }
        ]
    }
    
    metrics = {
        "total_grants_funded": 37,
        "total_funding_amount": 42500000,
        "average_grant_size": 1148648.65,
        "application_success_rate": 38.5,
        "citations_per_publication": 8.7
    }
    
    # Create the institution
    institution = Institution(
        name="Sheba Medical Center",
        location={
            "address": "Derech Sheba 2",
            "city": "Ramat Gan",
            "state": "Tel Aviv District",
            "country": "Israel",
            "coordinates": {
                "latitude": 32.0461,
                "longitude": 34.8516
            }
        },
        overview="Sheba Medical Center is Israel's largest medical center and a leading academic medical facility. "
                 "Founded in 1948, Sheba has been at the forefront of medical innovation and has been recognized "
                 "as one of the top 10 hospitals in the world. The center combines excellent clinical care with "
                 "cutting-edge research across a wide range of medical specialties.",
        founded_year=1948,
        institution_type="Hospital",
        size={
            "beds": 1900,
            "annual_patients": 1250000,
            "employees": 8500,
            "research_budget": 75000000,
            "grant_success_rate": 38.5
        },
        contact={
            "website": "https://eng.sheba.co.il/",
            "email": "info@sheba.health.gov.il",
            "phone": "+972-3-530-3030",
            "main_contact_person": "Dr. Yitshak Kreiss, Director"
        },
        research_focus_areas=research_focus_areas,
        departments=departments,
        researchers=researchers,
        past_grants=past_grants,
        infrastructure=infrastructure,
        publications=publications,
        metrics=metrics
    )
    
    return institution


def create_sample_grants() -> pd.DataFrame:
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
    
    return pd.DataFrame(grants_data)


# ========== MAIN APPLICATION ==========

def add_institutional_profile_to_app():
    """Function to add the institutional profile tab to the Streamlit app"""
    # Create the institution data
    institution = create_sheba_medical_center()
    
    # Create sample grants data
    grants_df = create_sample_grants()
    
    # Save sample data to JSON file that can be loaded by the main app
    try:
        institution.save_to_json("data/sheba_medical_center.json")
        grants_df.to_csv("data/healthcare_grants.csv", index=False)
    except Exception as e:
        st.warning(f"Could not save data files: {str(e)}")
    
    # Display the institutional profile
    institutional_profile_tab(institution, grants_df)


# If running this file directly, show a demo
if __name__ == "__main__":
    st.set_page_config(page_title="Institutional Profile Demo", layout="wide")
    st.title("Institutional Profile Demo")
    
    # Create the institution data
    institution = create_sheba_medical_center()
    
    # Create sample grants data
    grants_df = create_sample_grants()
    
    # Display the institutional profile
    institutional_profile_tab(institution, grants_df)