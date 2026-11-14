import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google import genai
from gtts import gTTS
import io
import urllib.parse
from analytics import (
    audit_and_clean_data,
    find_anomalies,
    calculate_profit_leak,
    full_statistical_profile,
    analyze_day_patterns,
    analyze_categories
)

st.set_page_config(page_title="VocalData 360° Analytics", page_icon="📊", layout="wide")

st.title("📊 VocalData: Automated Business Intelligence & Goal Solver")
st.write("Complete 360° data diagnosis, Profit Leak (₹) detection, Goal-Seeker reverse math, bilingual voice updates, and 1-click WhatsApp reporting.")

# Load secret API key if configured on Streamlit Cloud
default_key = st.secrets.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Controls & Language")
    currency = st.selectbox("💱 Currency Symbol:", ["₹", "$", "€", "£"])
    auto_clean = st.checkbox("🧹 Auto-Clean Dirty Data", value=True)
    voice_lang = st.radio("🗣️ Voice & Report Language:", ["English (Simple + Executive)", "हिन्दी (Hindi)"])
    st.divider()
    if default_key:
        api_key = default_key
        st.success("✅ AI Engine Connected!")
    else:
        api_key = st.text_input("Paste Google Gemini API Key:", type="password")
        st.caption("Get a free key at ://google.com")

# 1-Click Rich Retail Dataset or Custom File Upload
use_sample = st.checkbox("✨ Load Rich 21-Day Multi-Category Shop Dataset (1-Click Demo)")
uploaded_file = st.file_uploader("Or upload any business CSV file (Sales, Clinic, Cafe, E-Commerce)", type=["csv"])

raw_df = None
if use_sample:
    dates = pd.date_range(start="2026-10-01", periods=21, freq="D")
    categories = ["Electronics", "Apparel", "Home & Kitchen"] * 7
    ad_spend = [500, 520, 480, 600, 950, 1100, 1050, 510, 490, 530, 620, 1200, 1350, 1150, 500, 540, 470, 610, 1000, 1250, 1100]
    revenue = [4200, 4350, 3100, 5100, 8900, 11200, 10400, 4300, 2900, 4500, 5300, 14800, 13200, 11500, 4150, 4600, 2800, 5200, 9400, 12600, 10900]
    customers = [42, 44, 31, 51, 89, 112, 104, 43, 29, 45, 53, 148, 132, 115, 41, 46, 28, 52, 94, 126, 109]
    raw_df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Category": categories,
        "Revenue": revenue,
        "Ad_Spend": ad_spend,
        "Customers": customers
    })
elif uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

if raw_df is not None:
    df, health = audit_and_clean_data(raw_df, auto_clean=auto_clean)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.error("❌ No numeric columns found in this dataset.")
    else:
        selected_col = st.selectbox("🎯 Choose Your Primary Metric to Analyze:", numeric_cols)
        
        profile = full_statistical_profile(df, selected_col)
        outliers = find_anomalies(df, selected_col)
        leak_amount, slow_days_count = calculate_profit_leak(df, selected_col)
        day_insights = analyze_day_patterns(df, selected_col)
        cat_insights = analyze_categories(df, selected_col)

        # =====================================================================
        # TOP TRAFFIC-LIGHT HEALTH SIGNAL & 4 VISUAL SCORECARDS
        # =====================================================================
        st.divider()
        if profile['growth_pct'] > 5:
            st.success(f"🟢 **GROWING STRONG ({profile['growth_pct']:+.1f}%):** Recent **{selected_col}** is trending higher than earlier days!")
        elif profile['growth_pct'] < -5:
            st.error(f"🔴 **NEEDS ATTENTION ({profile['growth_pct']:+.1f}%):** Recent **{selected_col}** has dropped compared to earlier days.")
        else:
            st.info(f"🟡 **STEADY & STABLE ({profile['growth_pct']:+.1f}%):** **{selected_col}** is staying consistent.")

        c1, c2, c3, c4 = st.columns(4)
        best_day_name = day_insights["by_day"].loc[day_insights["by_day"]['Average'].idxmax(), 'Day_Name'] if day_insights else "N/A"
        worst_day_name = day_insights["by_day"].loc[day_insights["by_day"]['Average'].idxmin(), 'Day_Name'] if day_insights else "N/A"
        
        with c1:
            st.metric(f"💰 Total {selected_col}", f"{currency}{profile['total']:,.0f}", f"Daily Avg: {currency}{profile['mean']:,.0f}")
        with c2:
            st.metric("🏆 Superstar Day", f"{best_day_name}", f"Peak: {currency}{profile['max']:,.0f}")
        with c3:
            st.metric("🐢 Slowest Day", f"{worst_day_name}", f"Lowest: {currency}{profile['min']:,.0f}", delta_color="off")
        with c4:
            st.metric("💸 Missed Opportunity / Leak", f"{currency}{leak_amount:,.0f}", f"Across {slow_days_count} below-normal days", delta_color="inverse")

        # =====================================================================
        # 5 ORGANIZED DATA SCIENTIST & BUSINESS EXECUTION TABS
        # =====================================================================
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 1. Day Patterns & Trends",
            "🎯 2. Goal-Seeker Math Solver",
            "🔗 3. Drivers & Categories",
            "📊 4. Distribution & Outliers",
            "🩺 5. Data Health & Table"
        ])

        # --- TAB 1: DAY-OF-WEEK POWER BARS & TRENDS ---
        with tab1:
            if day_insights is not None:
                st.subheader("Which Days Make the Most Money? (Visual Power Gauges)")
                by_day = day_insights["by_day"].copy()
                by_type = day_insights["by_type"]
                max_avg = by_day['Average'].max()
                
                col_bars, col_chart = st.columns([1.1, 1.2])
                with col_bars:
                    for _, row in by_day.iterrows():
                        ratio = int((row['Average'] / max_avg) * 10) if max_avg > 0 else 0
                        blocks = "🟩" * ratio + "⬜" * (10 - ratio)
                        badge = " 🏆 BEST" if row['Average'] == max_avg else (" 🐢 SLOW" if row['Average'] == by_day['Average'].min() else "")
                        st.write(f"**{row['Day_Name']:<10}** | {blocks} | **{currency}{row['Average']:,.0f}**{badge}")
                        
                    if "Weekday (Mon-Fri)" in by_type and "Weekend (Sat-Sun)" in by_type:
                        wd, we = by_type["Weekday (Mon-Fri)"], by_type["Weekend (Sat-Sun)"]
                        lift = ((we - wd) / wd * 100) if wd != 0 else 0
                        st.info(f"📅 **Weekend vs. Weekday:** Weekends average **{currency}{we:,.0f}** vs. Weekdays **{currency}{wd:,.0f}** (**{lift:+.1f}%** lift on weekends).")
                        
                with col_chart:
                    fig_day_bar = px.bar(
                        by_day, x="Day_Name", y="Average", color="Average",
                        title=f"Average {selected_col} by Day of Week",
                        color_continuous_scale=["#EF4444", "#F59E0B", "#10B981"], text_auto=".0f"
                    )
                    st.plotly_chart(fig_day_bar, use_container_width=True)

            st.divider()
            st.subheader("Time-Series Trend + Smoothed Moving Average")
            df_trend = df.copy()
            window = min(5, max(2, len(df) // 4))
            df_trend[f"{window}-Day Moving Avg"] = df_trend[selected_col].rolling(window=window, min_periods=1).mean()
            x_axis = "Date" if "Date" in df_trend.columns else df_trend.index
            fig_line = px.line(df_trend, x=x_axis, y=[selected_col, f"{window}-Day Moving Avg"], markers=True)
            st.plotly_chart(fig_line, use_container_width=True)

        # --- TAB 2: GOAL-SEEKER REVERSE MATH SOLVER (UNIQUE DIFFERENTIATOR) ---
        with tab2:
            st.subheader(f"🎯 Goal-Seeker: How to Hit Your Next {selected_col} Target")
            st.write("Instead of guessing, enter the target number you want to reach. Python will calculate **3 exact mathematical paths** based on your real data:")
            
            default_target = int(profile['total'] * 1.25)
            target_goal = st.number_input(f"Enter your Target Total {selected_col} ({currency}):", min_value=1, value=max(10, default_target), step=1000)
            
            gap_needed = target_goal - profile['total']
            if gap_needed <= 0:
                st.success(f"🎉 You have already surpassed {currency}{target_goal:,.0f}! Try entering a higher target goal.")
            else:
                pct_boost_needed = (gap_needed / profile['total']) * 100
                extra_per_day = gap_needed / len(df)
                
                g1, g2, g3 = st.columns(3)
                with g1:
                    st.info(f"### Path A: Price / Ticket Lift\nRaise your average price or order value by **+{pct_boost_needed:.1f}%** (or add **{currency}{extra_per_day:,.0f}/day** in upsells) while keeping the exact same customers.")
                with g2:
                    if day_insights is not None:
                        slow_rows_total = by_day[by_day['Average'] < profile['mean']]['Total'].sum()
                        slow_boost = (gap_needed / slow_rows_total * 100) if slow_rows_total > 0 else pct_boost_needed
                        st.warning(f"### Path B: Fix Slow Days\nKeep your weekends as-is, and run a mid-week offer to boost your below-average days (like **{worst_day_name}**) by **+{slow_boost:.1f}%**.")
                    else:
                        st.warning(f"### Path B: Recover Profit Leak\nEliminating your **{currency}{leak_amount:,.0f}** leak on slow days immediately covers **{min(100, (leak_amount/gap_needed)*100):.0f}%** of your goal!")
                with g3:
                    if profile['top_driver'] and profile['driver_slope'] > 0:
                        extra_driver_units = gap_needed / (profile['driver_slope'] * len(df))
st.success(f"### Path C: Scale {profile['top_driver']}\nBased on your historical slope (+{profile['driver_slope']} per unit), increase {profile['top_driver']} by +{extra_driver_units:,.1f} units/day.")
else:
st.success(f"### Path C: Volume Growth\nAttract +{pct_boost_needed:.1f}% more daily customers across all days to reach {currency}{target_goal:,.0f}.")

        with tab3:
            st.subheader("What Actually Drives Growth? (Correlation & Categories)")
            if not profile['drivers_df'].empty:
                m1, m2 = st.columns(2)
                with m1:
                    corr_matrix = df[numeric_cols].corr().round(2)
                    fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix Heatmap")
                    st.plotly_chart(fig_heat, use_container_width=True)
                with m2:
                    top_feat = profile['top_driver']
                    st.info(f"🔗 **#1 Business Driver:** `{top_feat}` has a **{profile['top_corr']}** correlation with `{selected_col}`.")
                    fig_scat = px.scatter(df, x=top_feat, y=selected_col, title=f"{top_feat} vs. {selected_col}")
                    st.plotly_chart(fig_scat, use_container_width=True)
            else:
                st.info("Upload a dataset with multiple numeric columns (like Revenue + Ad_Spend) to unlock correlation driver analysis.")

# --- TAB 4: DISTRIBUTION & OUTLIERS ---
with tab4:
st.subheader(f"Statistical Distribution & Anomaly Audit: {selected_col}")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Mean (Average)", f"{currency}{profile['mean']:,.2f}")
s2.metric("Median (Typical)", f"{currency}{profile['median']:,.2f}")
s3.metric("Volatility (CV%)", f"{profile['cv']}%")
s4.metric("Skewness", f"{profile['skew']}")
d1, d2 = st.columns(2)
with d1:
fig_hist = px.histogram(df, x=selected_col, nbins=12, marginal="box", title="Spread & Boxplot")
st.plotly_chart(fig_hist, use_container_width=True)
with d2:
if len(outliers) > 0:
st.warning(f"🚨 Found {len(outliers)} statistical outliers:")
st.dataframe(outliers, use_container_width=True)
else:
st.success("✅ No extreme statistical outliers detected.")
# --- TAB 5: DATA HEALTH & RAW TABLE ---
with tab5:
st.subheader("Data Quality & Cleaning Report")
q1, q2, q3, q4 = st.columns(4)
q1.metric("Data Health Score", f"{health['health_score']}%")
q2.metric("Rows (Raw → Clean)", f"{health['raw_rows']} → {health['clean_rows']}")
q3.metric("Missing Cells Fixed", f"{health['missing_cells']}")
q4.metric("Duplicates Removed", f"{health['duplicate_rows']}")
st.write("#### Descriptive Summary Table")
st.dataframe(df.describe().round(2), use_container_width=True)
st.write("#### Full Dataset Table")
st.dataframe(df, use_container_width=True)
# =====================================================================
# ACTION CENTER: BILINGUAL VOICE BRIEF + 1-CLICK WHATSAPP DISPATCH
# =====================================================================
st.divider()
st.subheader("📲 Action Center: Bilingual Voice Update & 1-Click WhatsApp Report")
wa_summary = (
f"📊 VocalData Business Report ({selected_col})\n"
f"• Total: {currency}{profile['total']:,.0f} (Daily Avg: {currency}{profile['mean']:,.0f})\n"
f"• Growth Trend: {profile['growth_pct']:+.1f}%\n"
f"• 🏆 Best Day: {best_day_name} | 🐢 Slowest Day: {worst_day_name}\n"
f"• 💸 Missed Opportunity on Slow Days: {currency}{leak_amount:,.0f}\n"
f"• 🔗 Top Driver: {profile['top_driver'] or 'N/A'}"
)
wa_url = f"wa.me{urllib.parse.quote(wa_summary)}"
st.link_button("📲 Send Instant Summary to Owner's WhatsApp", wa_url, use_container_width=True)
if not api_key:
st.info("👈 Add your free Gemini API Key in Streamlit Secrets or the left sidebar to unlock spoken Hindi/English briefings!")
else:
client = genai.Client(api_key=api_key)
if st.button("▶️ Play Spoken Executive Summary (English / Hindi)", use_container_width=True):
with st.spinner("Generating spoken business briefing..."):
lang_prompt = "in clear, warm spoken Hindi (Devanagari script)" if "Hindi" in voice_lang else "in simple, direct English like a business advisor"
tts_code = "hi" if "Hindi" in voice_lang else "en"
prompt = f"""
Present this business audit {lang_prompt} in 4 short, punchy sentences:
1. State total {selected_col} ({currency}{profile['total']:,.0f}), daily average ({currency}{profile['mean']:,.0f}), and growth ({profile['growth_pct']:+.1f}%).
2. Highlight superstar day ({best_day_name}) vs slowest day ({worst_day_name}).
3. Mention the {currency}{leak_amount:,.0f} profit leak on slow days and #1 driver ({profile['top_driver']}).
4. Give one practical action tip for {worst_day_name}.
"""
resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
summary_txt = resp.text
st.success(summary_txt.replace("$", "$"))
tts = gTTS(text=summary_txt, lang=tts_code)
fp = io.BytesIO()
tts.write_to_fp(fp)
st.audio(fp, format="audio/mp3", autoplay=True)
st.download_button("📄 Download Complete Report (.txt)", data=f"{wa_summary}\n\nAI BRIEF:\n{summary_txt}", file_name="VocalData_Executive_Report.txt")
q = st.text_input("Ask any business question (e.g., 'What promotion should I run on slow days?'):")
if q:
with st.spinner("Analyzing..."):
q_resp = client.models.generate_content(
model="gemini-2.5-flash",
contents=f"Answer using ONLY these verified metrics: {wa_summary}. Keep it practical. Question: {q}"
)
st.info(q_resp.text.replace("$", "$"))
