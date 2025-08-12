# chart_page.py
import streamlit as st
import pandas as pd
import plotly.express as px

def render_chart_page(site_code):
    st.title(f"📈 Customer Revenue Analysis - {site_code}")

    # Ensure data is loaded from session
    if "official_data" not in st.session_state:
        st.warning("⚠️ Customer data not found. Generate the report first.")
        st.stop()

    df_raw = st.session_state["official_data"].copy()

    # Filter by selected site
    df_raw = df_raw[df_raw['Site'] == site_code]

    # Ensure Amount is numeric
    df_raw['Amount'] = pd.to_numeric(df_raw['Amount'], errors='coerce').fillna(0)

    # Create Period column (datetime)
    df_raw['Period'] = pd.to_datetime(df_raw['Year'].astype(str) + "-" + df_raw['Month'].astype(str).str.zfill(2), format="%Y-%m")

    # -------------------
    # Select Customers for charts
    customers = sorted(df_raw['Customer'].dropna().unique())
    default_customer = customers[0] if customers else None

    selected_customers = st.multiselect(
        "Select Customer(s) for Chart",
        customers,
        default=[default_customer] if default_customer else []
    )

    if not selected_customers:
        st.info("Select at least one customer.")
        st.stop()

    # -------------------
    # Line Chart
    st.markdown(f"### 📈 {', '.join(selected_customers)} - Line Chart")

    line_df = df_raw[df_raw['Customer'].isin(selected_customers)] \
        .groupby(['Customer', 'Period'], as_index=False)['Amount'].sum()

    fig_line = px.line(
        line_df,
        x='Period',
        y='Amount',
        color='Customer',
        title="Monthly Revenue Trend",
        markers=True
    )
    fig_line.update_layout(
        hovermode="x",
        hoverdistance=100,
        spikedistance=-1,
        xaxis=dict(
            showspikes=True,
            spikecolor="red",
            spikethickness=2,
            spikemode="across"
        ),
        hoverlabel=dict(
            bgcolor="black",
            font_size=14,
            font_color="white"
        )
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # -------------------
    # Bar Chart
    st.markdown(f"### 📊 {', '.join(selected_customers)} - Yearly Total Bar Chart")

    bar_df = df_raw[df_raw['Customer'].isin(selected_customers)] \
        .groupby(['Customer', 'Year'], as_index=False)['Amount'].sum()

    fig_bar = px.bar(
        bar_df,
        x='Year',
        y='Amount',
        color='Customer',
        title="Yearly Revenue",
        text_auto='.2s'
    )
    fig_bar.update_layout(
        xaxis_title="Year",
        yaxis_title="Total Amount (THB)"
    )
    st.plotly_chart(fig_bar, use_container_width=True)
