import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
from pathlib import Path

# Add project root directory to Python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from src.configs.params import DEFAULT_SETTINGS

# --- Page Config ---
st.set_page_config(
    page_title="오이 생장 시뮬레이션 대시보드",
    page_icon="🥒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Logger ---
# Configure logging for debugging within Streamlit
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# --- Helper Functions ---
def load_data(uploaded_file):
    """Load data from the uploaded Excel file, handling potential errors."""
    if uploaded_file is None:
        return None, None, None  # Return None if no file is uploaded

    try:
        # Read all sheets efficiently
        excel_data = pd.ExcelFile(uploaded_file)
        logger.info(f"Successfully opened Excel file: {uploaded_file.name}")
        logger.info(f"Sheets found: {excel_data.sheet_names}")

        daily_summary = None
        node_states = None
        params = DEFAULT_SETTINGS  # Initialize with default settings

        required_sheets = {
            "daily_summary",
            "Node_Final_States",
        }  # Remove 'parameters' from required sheets
        available_sheets = set(excel_data.sheet_names)

        missing_sheets = required_sheets - available_sheets
        if missing_sheets:
            st.error(f"오류: Excel 파일에 다음 필수 시트가 없습니다: {', '.join(missing_sheets)}")
            return None, None, None

        # --- Load Daily Summary ---
        try:
            daily_summary = pd.read_excel(excel_data, sheet_name="daily_summary")
            # Convert 'Date' column to datetime objects, handling potential errors
            try:
                daily_summary["Date"] = pd.to_datetime(daily_summary["Date"], errors="coerce")
                # Drop rows where date conversion failed
                original_count = len(daily_summary)
                daily_summary.dropna(subset=["Date"], inplace=True)
                if len(daily_summary) < original_count:
                    logger.warning(
                        f"Dropped {original_count - len(daily_summary)} rows due to invalid date format in 'daily_summary'."
                    )
                daily_summary.set_index("Date", inplace=True)
                logger.info("Successfully loaded and processed 'daily_summary' sheet.")
            except KeyError:
                st.error("오류: 'daily_summary' 시트에 'Date' 컬럼이 없습니다.")
                return None, None, None
            except Exception as e:
                st.error(f"오류: 'daily_summary' 시트의 'Date' 컬럼 처리 중 오류 발생: {e}")
                return None, None, None

        except Exception as e:
            st.error(f"오류: 'daily_summary' 시트를 읽는 중 오류 발생: {e}")
            return None, None, None  # Stop if critical sheet fails

        # --- Load Node Final States ---
        try:
            node_states = pd.read_excel(excel_data, sheet_name="Node_Final_States")
            # Convert date columns, coercing errors
            for col in ["AppearanceDate", "BudInitDate", "FloweringDate", "SetDate", "HarvestDate"]:
                if col in node_states.columns:
                    node_states[col] = pd.to_datetime(node_states[col], errors="coerce")
            logger.info("Successfully loaded 'Node_Final_States' sheet.")
        except Exception as e:
            st.error(f"오류: 'Node_Final_States' 시트를 읽는 중 오류 발생: {e}")
            # Allow proceeding without node_states if daily_summary is loaded
            node_states = None

        # --- Load Parameters (if available) ---
        try:
            if "parameters" in excel_data.sheet_names:
                # Parameters might be structured differently, assume key-value for now
                params_df = pd.read_excel(excel_data, sheet_name="parameters", header=None)
                # Attempt to create a dictionary, handling potential format issues
                if len(params_df.columns) >= 2:
                    # Merge with DEFAULT_SETTINGS, giving priority to Excel values
                    excel_params = pd.Series(params_df[1].values, index=params_df[0]).to_dict()
                    params.update(excel_params)
                    logger.info(
                        "Successfully loaded and merged 'parameters' sheet with DEFAULT_SETTINGS."
                    )
                else:
                    logger.warning(
                        "'parameters' sheet format not recognized, using DEFAULT_SETTINGS."
                    )
            else:
                logger.info("No 'parameters' sheet found, using DEFAULT_SETTINGS.")
        except Exception as e:
            logger.warning(f"Error loading 'parameters' sheet: {e}. Using DEFAULT_SETTINGS.")

        logger.info("Data loading complete.")
        return daily_summary, node_states, params

    except Exception as e:
        st.error(f"Excel 파일 처리 중 예상치 못한 오류 발생: {e}")
        logger.error(f"Unexpected error loading Excel file: {e}", exc_info=True)
        return None, None, None


# --- Sidebar ---
with st.sidebar:
    # --- Construct path using pathlib --- # MODIFIED
    logo_path = (
        Path(__file__).parent / "knu_logo.png"
    )  # 현재 파일(app.py)과 같은 디렉토리에 있는 knu_logo.png 경로 생성
    st.image(
        str(logo_path),  # Use the constructed path (convert Path object to string)
        width=150,
    )
    st.title("파일 업로드")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "시뮬레이션 결과 Excel 파일을 업로드하세요.", type=["xlsx"], accept_multiple_files=False
    )
    st.markdown("---")
    st.markdown(
        "**참고:** Dashboard는 `daily_summary`, `Node_Final_States`, `parameters` 시트가 포함된 Excel 파일을 예상합니다."
    )

# --- Main App Logic ---
if uploaded_file is not None:
    logger.info(f"File uploaded: {uploaded_file.name}, Size: {uploaded_file.size}")
    daily_df, node_df, params_dict = load_data(uploaded_file)

    if daily_df is not None:
        st.success(f"'{uploaded_file.name}' 파일 로드 성공!")

        # --- Date Selector --- # NEW
        available_dates = daily_df.index.to_list()
        selected_date = st.date_input(
            "분석 기준 날짜 선택",
            value=available_dates[-1],
            min_value=available_dates[0],
            max_value=available_dates[-1],
            format="YYYY-MM-DD",
        )
        # Convert selected_date back to Timestamp for indexing if needed (st.date_input returns datetime.date)
        # Let's ensure it aligns with the DataFrame index type
        selected_date_ts = pd.Timestamp(selected_date)

        # --- Display DataFrames (Optional for Debugging) ---
        # with st.expander("Raw Dataframes"):
        #     st.subheader("Daily Summary")
        #     st.dataframe(daily_df.head())
        #     if node_df is not None:
        #         st.subheader("Node Final States")
        #         st.dataframe(node_df.head())
        #     if params_dict:
        #         st.subheader("Parameters")
        #         st.json(params_dict, expanded=False)

        # --- Dashboard Tabs ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            [
                "📊 현재 상태",
                "📈 생육 예측",
                "🌱 광합성/자원",
                "☀️ 환경 영향",
                "🍈 결실 관리",
                "💡 관리 제안",  # New Tab
            ]
        )

        # --- Tab 1: Current Status --- MODIFIED
        with tab1:
            st.header("📊 현재 상태 요약")

            # Get the latest data row - MODIFIED to use selected date
            if selected_date_ts in daily_df.index:
                selected_data = daily_df.loc[selected_date_ts]
                st.subheader(f"기준 날짜: {selected_date.strftime('%Y-%m-%d')}")  # MODIFIED
            else:
                st.error(
                    f"{selected_date.strftime('%Y-%m-%d')}에 해당하는 데이터가 없습니다. 날짜를 다시 선택해주세요."
                )
                # Optionally, default back to latest or stop processing for this tab
                selected_data = daily_df.iloc[
                    -1
                ]  # Default to latest if selected is somehow invalid
                st.subheader(f"기준 날짜: {daily_df.index[-1].strftime('%Y-%m-%d')} (기본값)")

            # --- Display Key Metrics --- # MODIFIED to use selected_data
            col1, col2, col3, col4 = st.columns(4)

            required_metrics = {
                "total_nodes": "총 마디 수 (개)",
                "Nodes_State_Set": "착과된 과실 수 (개)",
                "Total_Fruit_DW (g/m^2)": "총 과실 무게 (g/m²)",
                "Source_SO (g/m^2/d)": "일일 건물 생산량 (g/m²/d)",
            }

            metrics_data = {}
            all_metrics_present_for_chart = True  # Renamed to avoid conflict
            for col_name, label in required_metrics.items():
                if col_name in selected_data:  # Use selected_data
                    metrics_data[col_name] = selected_data[col_name]  # Use selected_data
                else:
                    st.warning(
                        f"주의: 'daily_summary' 시트에 '{col_name}' 컬럼이 없어 현재 상태 지표를 표시할 수 없습니다."
                    )
                    metrics_data[col_name] = "N/A"
                    if col_name in [
                        "total_nodes",
                        "Nodes_State_Set",
                        "Total_Fruit_DW (g/m^2)",
                        "Source_SO (g/m^2/d)",
                    ]:
                        all_metrics_present_for_chart = False

            # Format and display metrics if possible
            def format_metric(value):
                if isinstance(value, (int, float)):
                    return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                return value  # Return N/A as is

            with col1:
                st.metric(
                    label=required_metrics["total_nodes"],
                    value=format_metric(metrics_data["total_nodes"]),
                )
            with col2:
                st.metric(
                    label=required_metrics["Nodes_State_Set"],
                    value=format_metric(metrics_data["Nodes_State_Set"]),
                )
            with col3:
                st.metric(
                    label=required_metrics["Total_Fruit_DW (g/m^2)"],
                    value=format_metric(metrics_data["Total_Fruit_DW (g/m^2)"]),
                )
            with col4:
                st.metric(
                    label=required_metrics["Source_SO (g/m^2/d)"],
                    value=format_metric(metrics_data["Source_SO (g/m^2/d)"]),
                )

            st.markdown("---")
            st.subheader("주요 생육 지표 추이")

            # --- Display Time Series Charts --- # MODIFIED to add vline
            if all_metrics_present_for_chart:
                # Create figure with secondary y-axis
                fig_status = make_subplots(specs=[[{"secondary_y": True}]])

                # Nodes and Fruits (Left Y-axis)
                fig_status.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["total_nodes"],
                        name="총 마디 수",
                        mode="lines",
                    ),
                    secondary_y=False,
                )
                fig_status.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Nodes_State_Set"],
                        name="착과된 과실 수",
                        mode="lines",
                    ),
                    secondary_y=False,
                )

                # Weight and Source (Right Y-axis)
                fig_status.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Total_Fruit_DW (g/m^2)"],
                        name="총 과실 무게",
                        mode="lines",
                    ),
                    secondary_y=True,
                )
                fig_status.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Source_SO (g/m^2/d)"],
                        name="일일 건물 생산량",
                        mode="lines",
                    ),
                    secondary_y=True,
                )

                # Update layout
                fig_status.update_layout(
                    title_text="주요 지표 변화",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_status.update_xaxes(title_text="날짜")
                fig_status.update_yaxes(title_text="개수", secondary_y=False)
                fig_status.update_yaxes(title_text="무게 또는 생산량 (g/m²)", secondary_y=True)

                # Add vertical line for selected date # NEW / MODIFIED (removed annotation)
                fig_status.add_vline(
                    x=selected_date_ts, line_width=2, line_dash="dash", line_color="grey"
                )

                st.plotly_chart(fig_status, use_container_width=True)
            else:
                st.info("차트를 표시하기 위한 모든 컬럼이 'daily_summary'에 존재하지 않습니다.")

        with tab2:
            st.header("📈 생육 단계 및 예측")
            st.write("생육 단계별 노드 수 및 수확 예측 정보가 여기에 표시됩니다.")

            # --- 1. Node State Distribution Over Time ---
            st.subheader("생육 단계별 노드 수 변화")

            node_state_cols = [
                "Nodes_State_Bud",
                "Nodes_State_Flowering",
                "Nodes_State_Set",
                "Nodes_State_Harvested",
                "Nodes_State_Failed",
                "Nodes_State_Thinned",
            ]
            # Check if all required columns exist
            missing_state_cols = [col for col in node_state_cols if col not in daily_df.columns]

            if not missing_state_cols:
                # Melt the dataframe for Plotly Express area chart
                df_melted = daily_df.reset_index().melt(
                    id_vars=["Date"],
                    value_vars=node_state_cols,
                    var_name="Node State",
                    value_name="Count",
                )

                # Define Korean labels for the legend
                state_labels_kr = {
                    "Nodes_State_Bud": "꽃눈",
                    "Nodes_State_Flowering": "개화",
                    "Nodes_State_Set": "착과",
                    "Nodes_State_Harvested": "수확됨",
                    "Nodes_State_Failed": "착과 실패",
                    "Nodes_State_Thinned": "적과됨",
                }
                df_melted["Node State Label"] = df_melted["Node State"].map(state_labels_kr)

                fig_node_states = px.area(
                    df_melted,
                    x="Date",
                    y="Count",
                    color="Node State Label",  # Use Korean labels for color legend
                    title="일별 노드 상태 분포",
                    labels={"Date": "날짜", "Count": "노드 수", "Node State Label": "노드 상태"},
                    # Optional: Define a specific color sequence
                    # color_discrete_map={...}
                )
                fig_node_states.update_layout(
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_node_states, use_container_width=True)
            else:
                st.warning(
                    f"주의: 노드 상태 분포 차트를 그리기 위한 다음 컬럼이 'daily_summary' 시트에 없습니다: {', '.join(missing_state_cols)}"
                )

            st.markdown("---")

            # --- 2. Harvest Prediction --- # MODIFIED SECTION
            st.subheader("일별 수확량 및 총 수확량")

            # Check for required columns in daily_summary for the new chart
            harvest_dw_today_col = "Harvested_Fruit_DW_Today (g/m^2)"

            if harvest_dw_today_col in daily_df.columns:
                # --- Daily Harvest Amount Chart --- # NEW
                st.write(f"'daily_summary' 시트의 '{harvest_dw_today_col}' 데이터를 사용합니다.")

                # Filter out days with zero harvest for clarity, unless all are zero
                daily_harvest_data = daily_df[daily_df[harvest_dw_today_col] > 0]

                if not daily_harvest_data.empty:
                    fig_daily_harvest = px.bar(
                        daily_harvest_data,
                        x=daily_harvest_data.index,
                        y=harvest_dw_today_col,
                        title="일별 수확된 과실 무게",
                        labels={harvest_dw_today_col: "일별 수확량 (g/m²)", "index": "날짜"},
                    )
                    fig_daily_harvest.update_layout(bargap=0.2)
                    st.plotly_chart(fig_daily_harvest, use_container_width=True)
                else:
                    st.info("시뮬레이션 기간 동안 기록된 일별 수확량이 없습니다.")

                # --- Calculate and display Total Harvested Weight --- # NEW
                total_harvested_weight_from_daily = daily_df[harvest_dw_today_col].sum()
                st.metric(
                    label=f"총 수확 무게 (g/m²) (daily_summary '{harvest_dw_today_col}' 합계)",
                    value=f"{total_harvested_weight_from_daily:,.2f}",
                )

            else:
                st.warning(
                    f"주의: 일별 수확량 차트 및 총 수확량 계산을 위해 'daily_summary' 시트에 '{harvest_dw_today_col}' 컬럼이 필요합니다."
                )

            # --- Previous Node_Final_States based chart info --- # KEPT AS INFO
            st.markdown("---")
            st.info(
                "참고: 'Node_Final_States' 시트에 개별 노드의 수확 날짜(예: HarvestDate) 정보가 없어, 노드별 예상 수확 시기 분포 차트는 표시할 수 없습니다."
            )
            # --- END OF MODIFIED SECTION ---

        with tab3:
            st.header("🌱 광합성 및 자원 배분")

            # --- Check required columns ---
            required_cols_tab3 = [
                "A_grossCH2O_PRODUCTION (g/m^2/day)",  # Placeholder, will be corrected below
                "Source_SO (g/m^2/d)",
                "Sink_SI (g/m^2/d)",
                "SO_SI_Ratio",
            ]
            # Correct the Gross Photosynthesis column name if necessary (common variation)
            gross_photo_col = None
            if "A_grossCH2O_PRODUCTION (g/m^2/day)" in daily_df.columns:
                gross_photo_col = "A_grossCH2O_PRODUCTION (g/m^2/day)"
            elif "A_grossCH2O_PRODUCTION" in daily_df.columns:  # Handle potential variations
                gross_photo_col = "A_grossCH2O_PRODUCTION"
            # If found, update the first element in required list for checking
            if gross_photo_col:
                required_cols_tab3[0] = gross_photo_col
            else:
                # If neither variation is found, mark it as missing for the check below
                required_cols_tab3.pop(0)  # Remove the placeholder

            missing_cols_tab3 = [col for col in required_cols_tab3 if col not in daily_df.columns]
            if gross_photo_col is None:
                missing_cols_tab3.append(
                    "'A_grossCH2O_PRODUCTION (g/m^2/day)' or 'A_grossCH2O_PRODUCTION'"
                )

            if not missing_cols_tab3:
                st.subheader("광합성 및 Source/Sink 추이")

                fig_ps_ss = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=("광합성 및 Source/Sink 양", "SO/SI 비율"),
                    vertical_spacing=0.1,
                )

                # --- Row 1: Photosynthesis, Source, Sink Amounts ---
                # Gross Photosynthesis
                fig_ps_ss.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df[gross_photo_col],
                        name="총 광합성량 (CH₂O)",
                        mode="lines",
                        line=dict(color="green"),
                    ),
                    row=1,
                    col=1,
                )
                # Source (Net DM Production)
                fig_ps_ss.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Source_SO (g/m^2/d)"],
                        name="Source (순생산량 DM)",
                        mode="lines",
                        line=dict(color="blue"),
                    ),
                    row=1,
                    col=1,
                )
                # Sink (Total Demand)
                fig_ps_ss.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Sink_SI (g/m^2/d)"],
                        name="Sink (총요구량 DM)",
                        mode="lines",
                        line=dict(color="red"),
                    ),
                    row=1,
                    col=1,
                )

                # --- Row 2: SO/SI Ratio ---
                fig_ps_ss.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["SO_SI_Ratio"],
                        name="SO/SI 비율",
                        mode="lines",
                        line=dict(color="purple"),
                    ),
                    row=2,
                    col=1,
                )
                # Add a horizontal line at SO/SI = 1.0 for reference
                fig_ps_ss.add_hline(
                    y=1.0,
                    line_dash="dash",
                    line_color="grey",
                    annotation_text="Balance (1.0)",
                    annotation_position="bottom right",
                    row=2,
                    col=1,
                )

                # --- Update Layout ---
                fig_ps_ss.update_layout(
                    height=600,  # Adjust height for two subplots
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                )
                # Update y-axis titles
                fig_ps_ss.update_yaxes(title_text="생산량/요구량 (g/m²/d)", row=1, col=1)
                fig_ps_ss.update_yaxes(title_text="비율", row=2, col=1)
                # Update x-axis title only on the bottom subplot
                fig_ps_ss.update_xaxes(title_text="날짜", row=2, col=1)

                st.plotly_chart(fig_ps_ss, use_container_width=True)

                # --- Optional: Partitioning Ratio (If data exists) ---
                st.subheader("자원 배분 비율 (추정)")
                part_fruit_col = "Partitioning to fruit ratio"
                part_veg_col = "Partitioning to vegetative ratio"
                if part_fruit_col in daily_df.columns and part_veg_col in daily_df.columns:
                    # Create stacked area chart for partitioning ratios
                    part_cols = [part_fruit_col, part_veg_col]
                    df_part_melted = daily_df.reset_index().melt(
                        id_vars=["Date"], value_vars=part_cols, var_name="Organ", value_name="Ratio"
                    )
                    # Map Organ names to Korean
                    organ_labels_kr = {
                        part_fruit_col: "과실 배분 비율",
                        part_veg_col: "영양생장 배분 비율",
                    }
                    df_part_melted["Organ Label"] = df_part_melted["Organ"].map(organ_labels_kr)

                    fig_partition = px.area(
                        df_part_melted,
                        x="Date",
                        y="Ratio",
                        color="Organ Label",
                        title="자원 배분 비율 변화",
                        labels={"Date": "날짜", "Ratio": "비율", "Organ Label": "배분 대상"},
                    )
                    fig_partition.update_layout(
                        hovermode="x unified",
                        yaxis_tickformat=".0%",  # Format y-axis as percentage
                        legend=dict(
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                        ),
                    )
                    st.plotly_chart(fig_partition, use_container_width=True)
                else:
                    st.info(
                        f"자원 배분 비율 데이터('{part_fruit_col}', '{part_veg_col}')가 'daily_summary'에 없어 차트를 표시할 수 없습니다."
                    )

            else:
                # Ensure the error message clearly lists what's missing
                missing_list_str = ", ".join([f"'{col}'" for col in missing_cols_tab3])
                st.warning(
                    f"주의: 광합성/자원 탭 차트를 그리기 위한 다음 컬럼이 'daily_summary' 시트에 없습니다: {missing_list_str}"
                )

        with tab4:
            st.header("☀️ 환경 영향 분석")
            st.write("주요 환경 요인 추이 및 스트레스 분석 결과가 여기에 표시됩니다.")

            # --- Check required columns ---
            required_cols_tab4 = ["Daily_Avg_Temp", "Daily_PAR_MJ"]
            missing_cols_tab4 = [col for col in required_cols_tab4 if col not in daily_df.columns]

            if not missing_cols_tab4:
                st.subheader("주요 환경 요인 추이")

                fig_env = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=("일 평균 온도", "일 누적 광량 (PAR)"),
                    vertical_spacing=0.1,
                )

                # --- Row 1: Temperature ---
                fig_env.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Daily_Avg_Temp"],
                        name="일 평균 온도",
                        mode="lines",
                        line=dict(color="orange"),
                    ),
                    row=1,
                    col=1,
                )

                # Add optimal temperature range if parameters exist
                temp_opt_min = params_dict.get("optimal_temp_min")
                temp_opt_max = params_dict.get("optimal_temp_max")
                try:
                    # Convert to float, handle None or non-numeric gracefully
                    temp_opt_min = float(temp_opt_min) if temp_opt_min is not None else None
                    temp_opt_max = float(temp_opt_max) if temp_opt_max is not None else None
                    if temp_opt_min is not None and temp_opt_max is not None:
                        fig_env.add_hrect(
                            y0=temp_opt_min,
                            y1=temp_opt_max,
                            line_width=0,
                            fillcolor="green",
                            opacity=0.1,
                            annotation_text=f"적정 온도 ({temp_opt_min}-{temp_opt_max}°C)",
                            annotation_position="bottom right",
                            row=1,
                            col=1,
                        )
                    elif temp_opt_min is not None:
                        fig_env.add_hline(
                            y=temp_opt_min,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"최소 적정 온도 ({temp_opt_min}°C)",
                            row=1,
                            col=1,
                        )
                    elif temp_opt_max is not None:
                        fig_env.add_hline(
                            y=temp_opt_max,
                            line_dash="dash",
                            line_color="green",
                            annotation_text=f"최대 적정 온도 ({temp_opt_max}°C)",
                            row=1,
                            col=1,
                        )
                except (ValueError, TypeError):
                    logger.warning("Could not parse optimal temperature parameters.")
                    st.caption(
                        "참고: 파라미터 시트에서 숫자 형식의 'optimal_temp_min' 또는 'optimal_temp_max'를 찾을 수 없어 온도 적정 범위를 표시할 수 없습니다."
                    )

                # --- Row 2: PAR ---
                fig_env.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Daily_PAR_MJ"],
                        name="일 누적 광량",
                        mode="lines",
                        line=dict(color="gold"),
                    ),
                    row=2,
                    col=1,
                )

                # Add optimal PAR range (Example - you might need different param names)
                par_opt_min = params_dict.get("optimal_par_min_mj")  # Example name
                par_opt_max = params_dict.get("optimal_par_max_mj")  # Example name
                try:
                    par_opt_min = float(par_opt_min) if par_opt_min is not None else None
                    par_opt_max = float(par_opt_max) if par_opt_max is not None else None
                    if par_opt_min is not None and par_opt_max is not None:
                        fig_env.add_hrect(
                            y0=par_opt_min,
                            y1=par_opt_max,
                            line_width=0,
                            fillcolor="yellow",
                            opacity=0.1,
                            annotation_text=f"적정 광량 ({par_opt_min}-{par_opt_max} MJ)",
                            annotation_position="bottom right",
                            row=2,
                            col=1,
                        )
                    # Add lines for min/max if only one is defined (similar to temp)
                except (ValueError, TypeError):
                    logger.warning("Could not parse optimal PAR parameters.")
                    # Optionally inform user if params are missing
                    # st.caption("참고: 파라미터 시트에서 광량 적정 범위 정보를 찾을 수 없습니다.")

                # --- Update Layout ---
                fig_env.update_layout(
                    height=600,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                )
                fig_env.update_yaxes(title_text="온도 (°C)", row=1, col=1)
                fig_env.update_yaxes(title_text="광량 (MJ/m²/d)", row=2, col=1)
                fig_env.update_xaxes(title_text="날짜", row=2, col=1)

                st.plotly_chart(fig_env, use_container_width=True)

                # --- Potential Stress Analysis (Example - Needs more logic) ---
                # st.subheader("환경 스트레스 분석 (예시)")
                # Define stress thresholds (could be from params)
                # high_temp_threshold = params_dict.get('high_temp_stress_threshold', 35)
                # low_temp_threshold = params_dict.get('low_temp_stress_threshold', 10)
                # low_par_threshold = params_dict.get('low_par_stress_threshold', 5)

                # high_temp_days = daily_df[daily_df['Daily_Avg_Temp'] > high_temp_threshold].index
                # low_temp_days = daily_df[daily_df['Daily_Avg_Temp'] < low_temp_threshold].index
                # low_par_days = daily_df[daily_df['Daily_PAR_MJ'] < low_par_threshold].index

                # if not high_temp_days.empty:
                #     st.warning(f"고온 스트레스 가능 기간 ({len(high_temp_days)}일): {high_temp_days.min().strftime('%Y-%m-%d')} ~ {high_temp_days.max().strftime('%Y-%m-%d')}")
                # if not low_temp_days.empty:
                #     st.warning(f"저온 스트레스 가능 기간 ({len(low_temp_days)}일): {low_temp_days.min().strftime('%Y-%m-%d')} ~ {low_temp_days.max().strftime('%Y-%m-%d')}")
                # if not low_par_days.empty:
                #     st.warning(f"광 부족 가능 기간 ({len(low_par_days)}일): {low_par_days.min().strftime('%Y-%m-%d')} ~ {low_par_days.max().strftime('%Y-%m-%d')}")
                # if high_temp_days.empty and low_temp_days.empty and low_par_days.empty:
                #     st.success("시뮬레이션 기간 동안 주요 환경 스트레스 요인이 감지되지 않았습니다.")

            else:
                st.warning(
                    f"주의: 환경 영향 탭 차트를 그리기 위한 다음 컬럼이 'daily_summary' 시트에 없습니다: {', '.join(missing_cols_tab4)}"
                )

        with tab5:
            st.header("🍈 결실 관리")

            # --- Check required columns ---
            required_cols_tab5 = [
                "Nodes_State_Flowering",
                "Nodes_State_Set",
                "Permanently_Failed_Nodes_Today",
                "Cumulative_Failed_Nodes",
                "Total_Fruit_DW (g/m^2)",
            ]
            missing_cols_tab5 = [col for col in required_cols_tab5 if col not in daily_df.columns]

            if not missing_cols_tab5:
                st.subheader("결실 단계 및 실패 추이")

                fig_fruit = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=("일별 결실 상태 변화", "누적 실패 및 평균 과중(추정)"),
                    vertical_spacing=0.1,
                    specs=[
                        [{"secondary_y": False}],
                        [{"secondary_y": True}],
                    ],  # Secondary axis for avg weight
                )

                # --- Row 1: Daily Fruit Set Status ---
                fig_fruit.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Nodes_State_Flowering"],
                        name="개화 중 노드 수",
                        mode="lines",
                        line=dict(color="#ffcc00"),
                    ),  # Yellowish
                    row=1,
                    col=1,
                )
                fig_fruit.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Nodes_State_Set"],
                        name="착과 노드 수",
                        mode="lines",
                        line=dict(color="#33cc33"),
                    ),  # Green
                    row=1,
                    col=1,
                )
                fig_fruit.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Permanently_Failed_Nodes_Today"],
                        name="당일 실패 노드 수",
                        mode="lines",
                        line=dict(color="#cc0000"),
                    ),  # Red
                    row=1,
                    col=1,
                )

                # --- Row 2: Cumulative Failures and Estimated Avg Weight ---
                # Cumulative Failed Nodes (Left Y-axis)
                fig_fruit.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Cumulative_Failed_Nodes"],
                        name="누적 실패 노드 수",
                        mode="lines",
                        line=dict(color="#cc0000"),
                    ),  # Red again
                    row=2,
                    col=1,
                    secondary_y=False,
                )

                # Estimated Average Fruit Weight (Right Y-axis)
                # Avoid division by zero
                daily_df["Estimated_Avg_Fruit_Weight"] = (
                    daily_df["Total_Fruit_DW (g/m^2)"]
                    / daily_df["Nodes_State_Set"].replace(0, pd.NA)
                ).fillna(
                    0
                )  # Fill NA/NaN with 0 after division

                fig_fruit.add_trace(
                    go.Scatter(
                        x=daily_df.index,
                        y=daily_df["Estimated_Avg_Fruit_Weight"],
                        name="평균 과중 (추정)",
                        mode="lines",
                        line=dict(color="#9966ff"),
                    ),  # Purple
                    row=2,
                    col=1,
                    secondary_y=True,
                )

                # --- Update Layout ---
                fig_fruit.update_layout(
                    height=600,
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                )
                fig_fruit.update_yaxes(title_text="노드 수", row=1, col=1)
                fig_fruit.update_yaxes(
                    title_text="누적 실패 노드 수", row=2, col=1, secondary_y=False
                )
                fig_fruit.update_yaxes(
                    title_text="평균 과중 (g, 추정치)", row=2, col=1, secondary_y=True
                )
                fig_fruit.update_xaxes(title_text="날짜", row=2, col=1)

                st.plotly_chart(fig_fruit, use_container_width=True)

                # Clean up temporary column
                # del daily_df['Estimated_Avg_Fruit_Weight']

            else:
                st.warning(
                    f"주의: 결실 관리 탭 차트를 그리기 위한 다음 컬럼이 'daily_summary' 시트에 없습니다: {', '.join(missing_cols_tab5)}"
                )

        # --- Tab 6: Management Suggestions --- NEW
        with tab6:
            st.header("💡 관리 제안")
            st.markdown(
                "**주의:** 본 제안은 시뮬레이션 결과를 바탕으로 한 일반적인 참고 사항이며, 실제 농장 관리에는 농장주의 경험과 전문가의 정확한 진단이 필요합니다."
            )
            st.markdown("---")

            suggestions = []
            n_days_check = 3  # Check trends over the last N days relative to selected date

            # --- Use selected_date_ts for analysis --- # MODIFIED
            # Find the index location of the selected date
            try:
                selected_date_index_loc = daily_df.index.get_loc(selected_date_ts)
            except KeyError:
                st.error(
                    f"선택된 날짜({selected_date_ts.strftime('%Y-%m-%d')})에 해당하는 데이터를 찾을 수 없습니다."
                )
                selected_date_index_loc = -1  # Indicate error or fallback

            # Ensure we have enough data *before* the selected date for trend analysis
            if selected_date_index_loc != -1 and selected_date_index_loc >= n_days_check - 1:
                # Get data up to and including the selected date, then take the last n_days_check
                # The iloc slice end is exclusive, so +1 is needed to include the selected date
                recent_data = daily_df.iloc[
                    selected_date_index_loc - n_days_check + 1 : selected_date_index_loc + 1
                ]
                st.info(
                    f"**{selected_date.strftime('%Y-%m-%d')}** 기준 이전 {n_days_check}일간의 데이터로 분석합니다."
                )  # Added info

                # --- Display suggestions with Gauges --- # MODIFIED SECTION
                st.subheader("진단 및 제안 사항")
                any_suggestion_made = False  # Flag to track if any suggestion is displayed

                # --- Use selected_date_ts for analysis --- # (Logic moved slightly for structure)
                if selected_date_index_loc != -1 and selected_date_index_loc >= n_days_check - 1:
                    recent_data = daily_df.iloc[
                        selected_date_index_loc - n_days_check + 1 : selected_date_index_loc + 1
                    ]
                    st.info(
                        f"**{selected_date.strftime('%Y-%m-%d')}** 기준 이전 {n_days_check}일간의 데이터로 분석합니다."
                    )

                    # --- 1. Check SO/SI Ratio --- # VISUALIZED
                    so_si_col = "SO_SI_Ratio"
                    so_si_threshold = 0.2
                    if so_si_col in recent_data.columns:
                        avg_so_si = recent_data[so_si_col].mean()
                        any_suggestion_made = True  # A check was performed

                        gauge_col, text_col = st.columns(
                            [1, 2]
                        )  # Create columns for gauge and text

                        with gauge_col:
                            fig_so_si = go.Figure(
                                go.Indicator(
                                    mode="gauge+number",
                                    value=avg_so_si,
                                    title={
                                        "text": f"SO/SI 비율<br>(최근 {n_days_check}일 평균)",
                                        "font": {"size": 16},
                                    },
                                    gauge={
                                        "axis": {
                                            "range": [0, 1.0],
                                            "tickwidth": 1,
                                            "tickcolor": "darkblue",
                                        },
                                        "bar": {
                                            "color": (
                                                "darkblue"
                                                if avg_so_si >= so_si_threshold
                                                else "red"
                                            )
                                        },
                                        "bgcolor": "white",
                                        "borderwidth": 2,
                                        "bordercolor": "gray",
                                        "steps": [
                                            {
                                                "range": [0, so_si_threshold],
                                                "color": "rgba(255, 0, 0, 0.3)",
                                            },
                                            {
                                                "range": [so_si_threshold, 1.0],
                                                "color": "rgba(0, 255, 0, 0.1)",
                                            },
                                        ],
                                        "threshold": {
                                            "line": {"color": "red", "width": 4},
                                            "thickness": 0.75,
                                            "value": so_si_threshold,
                                        },
                                    },
                                )
                            )
                            fig_so_si.update_layout(
                                height=250, margin={"t": 50, "b": 50, "l": 50, "r": 50}
                            )
                            st.plotly_chart(fig_so_si, use_container_width=True)

                        with text_col:
                            st.markdown("**진단: 작물 활력 (SO/SI 비율)**")
                            if avg_so_si < so_si_threshold:
                                st.warning(
                                    f"**🔴 활력 저하 우려:** 최근 {n_days_check}일 평균 SO/SI 비율 (**{avg_so_si:.2f}**)이 기준치({so_si_threshold})보다 낮습니다. "
                                    f"이는 에너지 생산량(Source)이 작물 요구량(Sink)에 비해 부족할 수 있음을 의미합니다."
                                )
                                st.markdown(
                                    "**제안:** 광합성 증진(광량, CO2 농도 확인 및 조절) 또는 Sink 요구량 조절(야간 온도 관리, 과도한 착과 여부 점검) 방안을 검토해 보세요."
                                )
                            else:
                                st.success(
                                    f"**🟢 양호:** 최근 {n_days_check}일 평균 SO/SI 비율 (**{avg_so_si:.2f}**)이 기준치({so_si_threshold}) 이상으로 유지되고 있습니다."
                                )

                    # --- 2. Check Fruit Failure Rate --- (Placeholder for visualization)
                    failure_col = "Permanently_Failed_Nodes_Today"
                    failure_threshold = params_dict.get("failure_rate_warning_threshold", 1.0)
                    if failure_col in recent_data.columns:
                        avg_failures = recent_data[failure_col].mean()
                        any_suggestion_made = True
                        # TODO: Add gauge chart and text columns similar to SO/SI
                        if avg_failures > failure_threshold:
                            st.warning(  # Keep original text for now
                                f"**착과 실패율 증가 ({selected_date.strftime('%m/%d')} 기준):** 최근 {n_days_check}일 평균 착과 실패 수 ({avg_failures:.1f}개/일)가 기준치({failure_threshold}개/일)보다 높습니다. "
                                f"결실 관리 탭에서 실패 추세를 확인하고, SO/SI 비율, 개화 후 기간, 환경 스트레스(온도, 광량) 등 관련 요인을 점검하여 원인을 파악해 보세요."
                            )
                        # else: Optionally add a success message

                    # --- 3. Check Temperature Deviation --- (Placeholder for visualization)
                    temp_col = "Daily_Avg_Temp"
                    temp_opt_min = params_dict.get("optimal_temp_min")
                    temp_opt_max = params_dict.get("optimal_temp_max")
                    if temp_col in recent_data.columns:
                        try:
                            temp_opt_min = float(temp_opt_min) if temp_opt_min is not None else None
                            temp_opt_max = float(temp_opt_max) if temp_opt_max is not None else None
                            avg_temp = recent_data[temp_col].mean()
                            any_suggestion_made = True
                            # TODO: Add gauge chart and text columns similar to SO/SI

                            if temp_opt_min is not None and avg_temp < temp_opt_min:
                                st.info(  # Keep original text for now
                                    f"**저온 경향 ({selected_date.strftime('%m/%d')} 기준):** 최근 {n_days_check}일 평균 온도 ({avg_temp:.1f}°C)가 설정된 적정 최저 온도({temp_opt_min}°C)보다 낮습니다. "
                                    f"지속적인 저온은 생육 지연 및 양분 흡수 저하를 유발할 수 있습니다. 야간 온도 관리 및 보온 상태를 점검하세요."
                                )
                            if temp_opt_max is not None and avg_temp > temp_opt_max:
                                st.info(  # Keep original text for now
                                    f"**고온 경향 ({selected_date.strftime('%m/%d')} 기준):** 최근 {n_days_check}일 평균 온도 ({avg_temp:.1f}°C)가 설정된 적정 최고 온도({temp_opt_max}°C)보다 높습니다. "
                                    f"지속적인 고온은 호흡량 증가, 광합성 효율 저하, 생리 장해를 유발할 수 있습니다. 환기, 차광, 냉방 시설 가동 상태를 점검하세요."
                                )
                            # else: Optionally add a success message if within range
                        except (ValueError, TypeError):
                            pass

                    # 4. Add more checks here (e.g., PAR levels, fruit growth rate)
                # --- Final message if no suggestions were made --- # MODIFIED
                elif (
                    not any_suggestion_made
                ):  # If data was available but no specific issues were flagged (or checks weren't applicable)
                    st.success(
                        f"{selected_date.strftime('%Y-%m-%d')} 기준 분석 결과, 특별히 우려되는 위험 신호는 감지되지 않았습니다. 현재 상태를 잘 유지하세요."
                    )
                # else: The case where there wasn't enough data is handled by the st.info above

            else:
                st.info(
                    f"분석 기준 날짜({selected_date.strftime('%Y-%m-%d')}) 이전에 충분한 데이터({n_days_check}일)가 없어 추세 기반 제안을 생성할 수 없습니다."
                )

    else:
        # Error messages are handled within load_data
        st.warning("데이터를 로드하지 못했습니다. 파일을 확인하고 다시 업로드해주세요.")
        logger.warning("Data loading failed, daily_df is None.")

else:
    st.info("대시보드를 보려면 사이드바에서 시뮬레이션 결과 파일을 업로드하세요.")
    logger.info("No file uploaded yet.")

# Add footer or additional info if needed
st.markdown("---")
st.caption("Cucumber Growth Simulation Dashboard v0.1")
