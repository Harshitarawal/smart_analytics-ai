import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from google import genai
from gtts import gTTS
import io
from analytics import find_anomalies

st.set_page_config(page_title="VocalData AI", page_icon="📊", layout="wide")

st.title("📊 VocalData: Your AI Data Assistant")
st.write("Upload your business data sheet below for automatic statistical audits, AI Q&A, voice briefs, and 'What-If' simulations.")

# Sidebar for entering your free Gemini API Key
with st.sidebar:
    st.header("🔑 AI Settings")
    api_key = st.text_input("Enter your free Google Gemini API Key:", type="password")
    st.caption("Get a free key at ://google.com")
    st.divider()
    st.markdown("### 🚀 Built With:")
    st.markdown("- **Deterministic Math:** Z-Score & IQR Outliers\n- **AI Engine:** Gemini 2.5 Flash\n- **Voice Brief:** Google TTS")

# 1-Click Sample Data Option + CSV Uploader
use_sample = st.checkbox("✨ Use built-in sample retail data (Great for quick demo!)")
uploaded_file = st.file_uploader("Or drop your own business CSV file here", type=["csv"])

df = None
if use_sample:
    df = pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Sales":[120,135,128,140,980,130,125],
        "Customers": [24, 27, 25, 29, 195, 26, 25]
    })
    st.info("📊 Loaded sample retail dataset (Notice Friday's massive spike!).")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

if df is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 1. Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
    with col2:
        st.write("### 2. Automatic Statistical Audit")
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        outliers = pd.DataFrame()
        
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Select a number column to audit:", numeric_cols)
            outliers = find_anomalies(df, selected_col)
            
            if len(outliers) > 0:
                st.warning(f"⚠️ Alert: Found {len(outliers)} unusual statistical anomalies in '{selected_col}'!")
                st.dataframe(outliers, use_container_width=True)
            else:
                st.info(f"✅ Everything looks normal in '{selected_col}'.")
                
            st.write("### 3. Visual Trend")
            fig = px.line(df, y=selected_col, title=f"Trend Overview: {selected_col}", markers=True)
            st.plotly_chart(fig, use_container_width=True)

    # Interactive "What-If" Math Simulator
    if len(numeric_cols) > 0:
        st.divider()
        st.write(f"### 4. Interactive 'What-If' Simulator ({selected_col})")
        st.write("Drag the sliders below to simulate how price or volume changes impact your numbers in real time:")
        
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            price_change = st.slider("Simulate Price / Rate Change (%)", -50, 100, 10)
        with sim_col2:
            volume_change = st.slider("Simulate Demand / Volume Change (%)", -50, 100, 5)
            
        multiplier = (1 + price_change / 100) * (1 + volume_change / 100)
        df_sim = df.copy()
        df_sim[f"Projected_{selected_col}"] = (df_sim[selected_col] * multiplier).round(2)
        
        baseline_total = df[selected_col].sum()
        projected_total = df_sim[f"Projected_{selected_col}"].sum()
        net_diff = projected_total - baseline_total
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Total", f"{baseline_total:,.2f}")
        m2.metric("Projected Total", f"{projected_total:,.2f}", f"{net_diff:+,.2f}")
        m3.metric("Net Growth Impact", f"{((multiplier - 1) * 100):+.1f}%")
        
        fig_sim = px.line(df_sim, y=[selected_col, f"Projected_{selected_col}"], 
                          title=f"Current vs. Projected {selected_col}", markers=True)
        st.plotly_chart(fig_sim, use_container_width=True)

    st.divider()
    
    # Ask Your Data & Executive Voice Brief
    st.write("### 5. Ask Your Data & Executive Voice Brief")
    
    if not api_key:
        st.info("👈 Paste your free Google Gemini API Key in the left sidebar to unlock AI Q&A and Voice Briefs!")
    else:
        client = genai.Client(api_key=api_key)
        
        data_context = f"""
        Columns: {list(df.columns)}
        Total Rows: {len(df)}
        Summary Statistics:\n{df.describe().to_string()}
        Detected Statistical Anomalies:\n{outliers.to_string() if not outliers.empty else 'None'}
        """
        
        if st.button("🎙️ Generate 30-Second Executive Voice Brief"):
            with st.spinner("Analyzing math and recording your audio brief..."):
                prompt = f"You are a senior business analyst. Based on this exact statistical data, write a crisp, friendly 3-sentence spoken summary highlighting key trends and any anomalies:\n{data_context}"
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                brief_text = response.text
                
                clean_display_text = brief_text.replace("$", "\\$")
                st.write("**Transcript:**", clean_display_text)
                
                tts = gTTS(text=brief_text, lang='en')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                st.audio(audio_fp, format='audio/mp3')
                
                report_content = f"VOCALDATA EXECUTIVE AUDIT REPORT\n{'='*35}\n\nAI SUMMARY:\n{brief_text}\n\nSTATISTICAL ANOMALIES DETECTED:\n{outliers.to_string() if not outliers.empty else 'None'}"
                st.download_button("📄 Download Executive Audit Report (.txt)", data=report_content, file_name="VocalData_Executive_Report.txt")

        user_question = st.text_input("Ask a question about your data (e.g., 'Why did Friday spike and what should I do?'):")
        if user_question:
            with st.spinner("Thinking..."):
                qa_prompt = f"Answer this business question using ONLY the verified data statistics below. Be clear and actionable.\nData:\n{data_context}\nQuestion: {user_question}"
                qa_response = client.models.generate_content(model="gemini-2.5-flash", contents=qa_prompt)
                st.write("#### 💡 AI Insight:")
                st.write(qa_response.text.replace("$", "\\$"))
