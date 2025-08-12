# chart_page.py (for streamlit-monthly-revenue)
import streamlit as st
import pandas as pd
import plotly.express as px

def render_chart_page(site_code):
    st.title(f"📊 Monthly Revenue Analysis - {site_code}")

    if "official_data" not in st.session_state:
        st.warning("⚠️ Data not found. Please generate the report first.")
        st.stop()

    df_raw = st.session_state["official_data"].copy()
    df_raw = df_raw[df_raw['Site'] == site_code]
    df_raw['Amount'] = pd.to_numeric(df_raw['Amount'], errors='coerce').fillna(0)
    df_raw['Period'] = pd.to_datetime(df_raw['Year'] + "-" + df_raw['Month'], format="%Y-%m")

    # --- Line Chart ---
    customers = sorted(df_raw['Customer'].dropna().unique())
    selected_customers = st.multiselect(
        "Select Customer Chart",
        customers,
        default=[customers[0]] if customers else []
    )
    if not selected_customers:
        st.info("Select at least one customer.")
        st.stop()

    selected_customers_display = [cust.split("]-", 1)[-1] for cust in selected_customers]
    st.markdown(f"### 📈 {', '.join(selected_customers_display)} - Line Chart")

    line_df = (
        df_raw[df_raw['Customer'].isin(selected_customers)]
        .groupby(['Customer', 'Period'], as_index=False)['Amount']
        .sum()
    )

    fig_line = px.line(
        line_df,
        x='Period',
        y='Amount',
        color='Customer',
        title="Monthly",
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

    # --- Bar Chart ---
    st.markdown(f"### 📊 {', '.join(selected_customers_display)} - Bar Chart")
    bar_df = (
        df_raw[df_raw['Customer'].isin(selected_customers)]
        .groupby(['Year'], as_index=False)['Amount']
        .sum()
        .sort_values(by='Amount', ascending=False)
    )

    fig_bar = px.bar(
        bar_df,
        x='Year',
        y='Amount',
        title="Yearly",
        text_auto='.2s'
    )
    fig_bar.update_layout(xaxis_title="", yaxis_title="Total Amount (THB)")
    st.plotly_chart(fig_bar, use_container_width=True)
