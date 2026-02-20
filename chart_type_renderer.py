import altair as alt
import streamlit as st


def show_chart_type_selector(key: str):
    """
    Displays horizontal chart type selector.
    """
    return st.radio(
        "Chart Type",
        ["Bar Chart", "Line Chart", "Pie Chart"],
        horizontal=True,
        key=f"{key}-chart-type",
    )


def render_chart(
    df,
    x_col,
    y_col,
    tooltip,
    domain_order=None,
    forecast_col=None,
    height=430,
):
    """
    Renders chart based on selected chart type.
    """

    chart_type = show_chart_type_selector(x_col)

    # ---------------- BAR ----------------
    if chart_type == "Bar Chart":

        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X(
                    f"{x_col}:N",
                    sort=domain_order,
                ),
                y=alt.Y(f"{y_col}:Q"),
                color=(
                    alt.condition(
                        alt.datum[forecast_col],
                        alt.value("#9CA3AF"),
                        alt.value("#1f77b4"),
                    )
                    if forecast_col
                    else alt.value("#1f77b4")
                ),
                tooltip=tooltip,
            )
        )

    # ---------------- LINE ----------------
    elif chart_type == "Line Chart":

        chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(
                    f"{x_col}:N",
                    sort=domain_order,
                ),
                y=alt.Y(f"{y_col}:Q"),
                tooltip=tooltip,
            )
        )

    # ---------------- PIE ----------------
    else:

        chart = (
            alt.Chart(df)
            .mark_arc(innerRadius=60)
            .encode(
                theta=alt.Theta(f"{y_col}:Q"),
                color=alt.Color(f"{x_col}:N"),
                tooltip=tooltip,
            )
        )

    st.altair_chart(chart.properties(height=height), use_container_width=True)
