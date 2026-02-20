import altair as alt
import streamlit as st
import pandas as pd


def render_chart_type_selector(key_prefix: str) -> str:
    """
    Renders chart type selector.
    Returns selected chart type.
    """

    chart_type = st.radio(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart"],
        horizontal=True,
        key=f"{key_prefix}-chart-type"
    )

    return chart_type


def render_dynamic_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    tooltip_cols: list,
    domain_order: list,
    title: str = "",
    is_forecast_col: str | None = None,
):
    """
    Renders selected chart type dynamically.
    """

    chart_type = render_chart_type_selector(title)

    if chart_type == "Bar Chart":

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{x_col}:N",
                    sort=domain_order,
                    title="",
                ),
                y=alt.Y(f"{y_col}:Q", title=""),
                color=(
                    alt.condition(
                        alt.datum[is_forecast_col],
                        alt.value("#9CA3AF"),
                        alt.value("#1f77b4"),
                    )
                    if is_forecast_col
                    else alt.value("#1f77b4")
                ),
                tooltip=tooltip_cols,
            )
        )

    elif chart_type == "Line Chart":

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    f"{x_col}:N",
                    sort=domain_order,
                    title="",
                ),
                y=alt.Y(f"{y_col}:Q", title=""),
                tooltip=tooltip_cols,
            )
        )

    else:  # Pie Chart

        chart = (
            alt.Chart(df)
            .mark_arc(innerRadius=50)
            .encode(
                theta=alt.Theta(f"{y_col}:Q"),
                color=alt.Color(f"{x_col}:N"),
                tooltip=tooltip_cols,
            )
        )

    st.altair_chart(chart.properties(height=430), use_container_width=True)
