import streamlit as st
import pandas as pd
import altair as alt
import json
import random
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing import List

# --- AI LIBRARIES ---
# We use try/except to prevent the app from crashing if a user hasn't installed a specific lib
try:
    import google.generativeai as genai
except ImportError:
    genai = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
try:
    import anthropic
except ImportError:
    anthropic = None
try:
    from groq import Groq
except ImportError:
    Groq = None

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VoC & Churn Intelligence",
    page_icon="🗣️",
    layout="wide"
)

# --- DATA STRUCTURES ---
class TicketAnalysis(BaseModel):
    ticket_id: str
    sentiment_score: float = Field(description="-1.0 (Negative) to 1.0 (Positive)")
    primary_topic: str = Field(description="E.g., Bug, Pricing, UX, Feature Request")
    churn_risk: str = Field(description="Low, Medium, High")
    summary: str = Field(description="3-word summary of the issue")

class BatchAnalysis(BaseModel):
    results: List[TicketAnalysis]
    strategic_insight: str = Field(description="Executive summary of the top issue facing the company")

# --- MOCK DATA GENERATOR (Unchanged) ---
def generate_mock_data(rows=10):
    issues = [
        ("2FA email delayed", "Bug", "The verification email arrives 5–10 minutes late, blocking login.", "High"),
        ("Password reset link expired", "Bug", "Reset links expire too quickly and force multiple retries.", "Medium"),
        ("SSO setup unclear", "UX", "The SSO configuration guide is confusing and incomplete.", "Medium"),
        ("Role permissions too coarse", "Feature Request", "We need more granular user permissions.", "Medium"),
        ("Audit logs missing fields", "Bug", "Audit logs do not show who made the change.", "High"),
        ("Slow dashboard load", "Performance", "Initial dashboard load takes over 8 seconds.", "High"),
        ("Search results inaccurate", "Bug", "Global search returns unrelated records.", "Medium"),
        ("Export CSV malformed", "Bug", "Exported CSV files have broken columns.", "High"),
        ("No PDF export", "Feature Request", "Executives require PDF reports for reviews.", "Low"),
        ("Billing cycle unclear", "Billing", "It’s not clear when usage resets each month.", "Medium"),
        ("Unexpected overage charges", "Billing", "We were charged overages without warning.", "High"),
        ("Usage metrics delayed", "Bug", "Usage data lags by several hours.", "Medium"),
        ("API rate limits undocumented", "Technical", "Rate limits are enforced but not documented.", "Medium"),
        ("Webhook retries missing", "Bug", "Failed webhooks are not retried automatically.", "High"),
        ("Integration setup too manual", "UX", "Integrations require too many manual steps.", "Medium"),
        ("CRM sync duplicates records", "Bug", "Customer records are duplicated after sync.", "High"),
        ("Mobile layout broken", "UX", "Tables overflow on mobile screens.", "Medium"),
        ("Dark mode missing", "Feature Request", "Dark mode would reduce eye strain.", "Low"),
        ("Notifications too noisy", "UX", "Too many alerts for low-priority events.", "Medium"),
        ("Critical alerts delayed", "Bug", "High-severity alerts arrive too late.", "High"),
        ("No sandbox environment", "Feature Request", "We need a sandbox for testing changes.", "Medium"),
        ("API auth token expires silently", "Bug", "Tokens expire without warning, breaking jobs.", "High"),
        ("Retry logic missing in SDK", "Technical", "SDK lacks built-in retry and backoff.", "Medium"),
        ("Billing invoice lacks detail", "Billing", "Invoices don’t show per-feature charges.", "Medium"),
        ("Contract terms not visible", "UX", "We can’t view our contract terms in-app.", "Low"),
        ("Feature rollout not communicated", "Service", "New features appear without notice.", "Low"),
        ("Release notes incomplete", "Service", "Release notes lack breaking-change details.", "Medium"),
        ("Performance degrades at scale", "Performance", "System slows noticeably past 50k records.", "High"),
        ("Bulk upload fails silently", "Bug", "Bulk uploads fail without error messages.", "High"),
        ("Error messages too generic", "UX", "Errors don’t explain how to fix the issue.", "Medium"),
        ("Timezone handling incorrect", "Bug", "Reports show incorrect timestamps.", "Medium"),
        ("No data retention controls", "Feature Request", "We need configurable data retention.", "Medium"),
        ("Compliance docs outdated", "Service", "SOC2 documentation is out of date.", "High"),
        ("Access revoked too slowly", "Security", "User access revocation is not immediate.", "High"),
        ("No IP allowlisting", "Security", "We require IP allowlisting for compliance.", "High"),
        ("Session timeout too aggressive", "UX", "Sessions expire too frequently.", "Low"),
        ("Session timeout too lenient", "Security", "Sessions remain active too long.", "Medium"),
        ("File size limit unclear", "UX", "Upload limits are not documented.", "Low"),
        ("File uploads slow", "Performance", "Large file uploads are extremely slow.", "Medium"),
        ("Data refresh manual only", "Feature Request", "Automatic data refresh would save time.", "Low"),
        ("Dashboard widgets inflexible", "Feature Request", "Widgets can’t be resized or rearranged.", "Low"),
        ("Pricing tiers confusing", "Pricing", "Differences between tiers are unclear.", "Medium"),
        ("Enterprise pricing opaque", "Pricing", "Enterprise pricing lacks transparency.", "Medium"),
        ("No usage forecast", "Feature Request", "Forecasting future usage would help budgeting.", "Low"),
        ("Renewal reminders missing", "Billing", "No reminder before contract renewal.", "High"),
        ("Cancel flow hard to find", "UX", "Cancellation requires contacting support.", "Medium"),
        ("Support response slow", "Service", "Took over 48 hours to get a response.", "High"),
        ("Support resolution excellent", "Service", "Issue resolved quickly and professionally.", "Low"),
        ("Onboarding too technical", "UX", "Non-technical users struggle to onboard.", "Medium"),
        ("Docs assume expert knowledge", "UX", "Documentation assumes prior system knowledge.", "Medium"),
        ("Sample data missing", "UX", "No sample data makes evaluation harder.", "Low"),
        ("Metrics definitions unclear", "UX", "KPIs lack clear definitions.", "Medium"),
        ("Inconsistent terminology", "UX", "Different pages use different terms.", "Low"),
        ("Multi-region support missing", "Feature Request", "We need regional data residency.", "High"),
        ("Latency high outside US", "Performance", "Response times are slow outside the US.", "High"),
        ("Backup restore unclear", "Service", "Restore process is not documented.", "Medium"),
        ("No SLA visibility", "Service", "SLA uptime metrics aren’t visible.", "Medium"),
        ("Changelog RSS missing", "Feature Request", "RSS feed for changes would help.", "Low"),
        ("Email notifications delayed", "Bug", "Notification emails arrive hours late.", "Medium"),
        ("Webhook payload inconsistent", "Bug", "Payload schema changes without notice.", "High"),
        ("SDK versioning unclear", "Technical", "Breaking changes are not versioned properly.", "High"),
        ("Data export throttled", "Performance", "Large exports are throttled excessively.", "Medium"),
        ("UI freezes on filters", "Bug", "Applying filters freezes the page.", "High"),
        ("No keyboard shortcuts", "Feature Request", "Power users need keyboard shortcuts.", "Low"),
        ("Great value for enterprise", "Pricing", "Delivers strong value at enterprise scale.", "Low"),
        ("Small team plan too limited", "Pricing", "Lower tiers miss essential features.", "Medium"),
        ("Feature request intake unclear", "Service", "No clear process to submit feature ideas.", "Low"),
        ("Roadmap visibility requested", "Service", "We’d like visibility into the roadmap.", "Low"),
        ("Data ownership unclear", "Legal", "Data ownership terms need clarification.", "High"),
        ("Contract export missing", "UX", "We can’t download signed contracts.", "Medium"),
        ("GDPR data deletion unclear", "Legal", "Process for requesting data deletion is not clear.", "High"),
        ("CCPA opt-out difficult", "Legal", "Customers find it hard to opt-out of data sales.", "High"),
        ("SOC2 report not public", "Security", "Need public access to SOC2 Type 2 report.", "Medium"),
        ("ISO 27001 certification missing", "Security", "Requires ISO 27001 for vendor approval.", "High"),
        ("Data residency options limited", "Compliance", "Need data to reside in specific regions (e.g., EU).", "High"),
        ("Vendor security questionnaire too long", "Sales", "Security questionnaire is excessively long and complex.", "Low"),
        ("No BAA available", "Legal", "Business Associate Agreement (BAA) is not provided.", "High"),
        ("Data encryption at rest missing", "Security", "Data is not encrypted at rest.", "High"),
        ("MFA not enforced", "Security", "Multi-factor authentication is optional, not enforced.", "High"),
        ("Password policy weak", "Security", "Password requirements are too lenient.", "Medium"),
        ("Access review process manual", "Security", "User access reviews are not automated.", "Medium"),
        ("Incident response plan unclear", "Security", "No clear documentation on incident response.", "High"),
        ("Penetration test reports unavailable", "Security", "Cannot access recent penetration test reports.", "High"),
        ("Sub-processor list outdated", "Compliance", "List of sub-processors is not current.", "Medium"),
        ("Data processing agreement generic", "Legal", "DPA is not tailored to specific needs.", "Medium"),
        ("No data export format options", "UX", "Data can only be exported in one format.", "Low"),
        ("API documentation outdated", "Technical", "API docs do not reflect current endpoints.", "Medium"),
        ("SDK examples missing", "Technical", "SDK lacks practical code examples.", "Low"),
        ("Error logging insufficient", "Technical", "Application error logs are not detailed enough.", "Medium"),
        ("Monitoring alerts too noisy", "Technical", "Too many false positive alerts from monitoring.", "Medium"),
        ("Deployment process complex", "Technical", "Deploying updates is a multi-step manual process.", "Medium"),
        ("No dark mode", "Feature Request", "Dark mode would reduce eye strain.", "Low"),
        ("Notifications too noisy", "UX", "Too many alerts for low-priority events.", "Medium"),
        ("Critical alerts delayed", "Bug", "High-severity alerts arrive too late.", "High"),
        ("No sandbox environment", "Feature Request", "We need a sandbox for testing changes.", "Medium"),
        ("API auth token expires silently", "Bug", "Tokens expire without warning, breaking jobs.", "High"),
        ("Retry logic missing in SDK", "Technical", "SDK lacks built-in retry and backoff.", "Medium"),
        ("Billing invoice lacks detail", "Billing", "Invoices don’t show per-feature charges.", "Medium"),
        ("Contract terms not visible", "UX", "We can’t view our contract terms in-app.", "Low"),
        ("Feature rollout not communicated", "Service", "New features appear without notice.", "Low"),
        ("Release notes incomplete", "Service", "Release notes lack breaking-change details.", "Medium"),
        ("Performance degrades at scale", "Performance", "System slows noticeably past 50k records.", "High"),
        ("Bulk upload fails silently", "Bug", "Bulk uploads fail without error messages.", "High"),
        ("Error messages too generic", "UX", "Errors don’t explain how to fix the issue.", "Medium"),
        ("Timezone handling incorrect", "Bug", "Reports show incorrect timestamps.", "Medium"),
        ("No data retention controls", "Feature Request", "We need configurable data retention.", "Medium"),
        ("Compliance docs outdated", "Service", "SOC2 documentation is out of date.", "High"),
        ("Access revoked too slowly", "Security", "User access revocation is not immediate.", "High"),
        ("No IP allowlisting", "Security", "We require IP allowlisting for compliance.", "High"),
        ("Session timeout too aggressive", "UX", "Sessions expire too frequently.", "Low"),
        ("Session timeout too lenient", "Security", "Sessions remain active too long.", "Medium"),
        ("File size limit unclear", "UX", "Upload limits are not documented.", "Low"),
        ("File uploads slow", "Performance", "Large file uploads are extremely slow.", "Medium"),
    ]
 
    data = []
    for i in range(rows):
        selected_issue = random.choice(issues)
        issue_type, topic, description, risk = selected_issue
        data.append({
            "Ticket ID": f"TICKET-{i+1:04d}",
            "Customer": f"CUST-{random.randint(1000, 9999)}",
            "Message": description,
            "Date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "priority": random.choice(["Low", "Medium", "High"]),
        })
    return pd.DataFrame(data)

# --- AI LOGIC ---
def analyze_batch(provider, api_key, df):
    # Convert dataframe to a text blob
    text_blob = df.to_json(orient="records")
    
    # Universal System Prompt
    system_prompt = """
    You are a Chief Product Officer (CPO) AI. 
    Analyze this batch of customer support tickets.
    1. Score Sentiment (-1 to 1).
    2. Categorize the topic strictly.
    3. Predict Churn Risk based on anger/frustration levels.
    4. Provide a Strategic Insight for the CEO.
    
    Return pure JSON adhering to this schema:
    {
        "results": [
            {"ticket_id": "...", "sentiment_score": 0.5, "primary_topic": "...", "churn_risk": "...", "summary": "..."}
        ],
        "strategic_insight": "..."
    }
    """

    try:
        # 1. GOOGLE GEMINI (AI Studio)
        if provider == "Google Gemini":
            if not genai: st.error("Library `google-generativeai` not installed."); return None
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash', 
                generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(f"{system_prompt}\n\nDATA:\n{text_blob}")
            return BatchAnalysis(**json.loads(response.text))

        # 2. OPENAI (GPT-4o)
        elif provider == "OpenAI":
            if not OpenAI: st.error("Library `openai` not installed."); return None
            client = OpenAI(api_key=api_key)
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text_blob}
                ],
                response_format=BatchAnalysis
            )
            return completion.choices[0].message.parsed

        # 3. ANTHROPIC (Claude 3.5 Sonnet)
        elif provider == "Anthropic (Claude)":
            if not anthropic: st.error("Library `anthropic` not installed."); return None
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": text_blob}]
            )
            # Claude returns JSON in text block, we parse manually
            return BatchAnalysis(**json.loads(message.content[0].text))

        # 4. GROQ (Llama 3 / Mixtral)
        elif provider == "Groq (Llama 3)":
            if not Groq: st.error("Library `groq` not installed."); return None
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": system_prompt + " RETURN ONLY JSON."},
                    {"role": "user", "content": text_blob}
                ],
                response_format={"type": "json_object"}
            )
            return BatchAnalysis(**json.loads(completion.choices[0].message.content))

    except Exception as e:
        st.error(f"Analysis Failed: {e}")
        return None

# --- UI LAYOUT ---

with st.sidebar:
    st.title("🗣️ InsightFlow AI")
    st.caption("BYO-Key Multi-Model Intelligence")
    st.divider()
    
    # Provider Selector
    provider = st.radio(
        "Select AI Model", 
        ["OpenAI", "Google Gemini", "Anthropic (Claude)", "Groq (Llama 3)"]
    )
    
    api_key = st.text_input(f"Enter {provider} API Key", type="password")
    
    st.info(f"Using: **{provider}**")
    
    st.markdown("---")
    if st.button("🎲 Get Mock Data"):
        st.session_state['data'] = generate_mock_data(20) # Lower count for faster demos
        st.success("Loaded tickets!")

# MAIN PAGE
st.title("Voice of Customer Intelligence")
st.markdown(f"Transform raw support tickets into insights using **{provider}**.")

# 1. DATA INPUT SECTION
if 'data' not in st.session_state:
    st.info("👈 Click 'Get Mock Data' in the sidebar to start.")
else:
    df = st.session_state['data']
    with st.expander("📄 Review Raw Data (CSV)", expanded=False):
        st.dataframe(df, use_container_width=True)

    if st.button("🚀 Analyze Batch", type="primary"):
        if not api_key:
            st.warning("⚠️ Please provide an API Key in the sidebar.")
        else:
            with st.spinner(f"Sending data to {provider}..."):
                analysis = analyze_batch(provider, api_key, df)
                
                if analysis:
                    # MERGE RESULTS
                    results_df = pd.DataFrame([item.model_dump() for item in analysis.results])
                    
                    # DASHBOARD
                    st.divider()
                    st.subheader("📊 Executive Dashboard")
                    
                    # Top Metric Cards
                    col1, col2, col3 = st.columns(3)
                    avg_sentiment = results_df['sentiment_score'].mean()
                    high_risk = results_df[results_df['churn_risk'] == 'High'].shape[0]
                    top_topic = results_df['primary_topic'].mode()[0] if not results_df.empty else "N/A"
                    
                    col1.metric("Customer Happiness (CSAT)", f"{avg_sentiment:.2f}")
                    col2.metric("High Churn Risk", str(high_risk), delta="Action Needed", delta_color="inverse")
                    col3.metric("Top Complaint", top_topic)
                    
                    # STRATEGIC INSIGHT
                    st.success(f"**AI Insight:** {analysis.strategic_insight}")
                    
                    # CHARTS ROW
                    c_chart1, c_chart2 = st.columns(2)
                    
                    with c_chart1:
                        st.write("**Topic Distribution**")
                        chart = alt.Chart(results_df).mark_bar().encode(
                            x='count()',
                            y=alt.Y('primary_topic', sort='-x'),
                            color='primary_topic'
                        )
                        st.altair_chart(chart, use_container_width=True)
                        
                    with c_chart2:
                        st.write("**Risk Matrix**")
                        scatter = alt.Chart(results_df).mark_circle(size=100).encode(
                            x='sentiment_score',
                            y='churn_risk',
                            color=alt.Color('churn_risk', scale=alt.Scale(domain=['Low', 'Medium', 'High'], range=['green', 'orange', 'red'])),
                            tooltip=['ticket_id', 'summary']
                        ).interactive()
                        st.altair_chart(scatter, use_container_width=True)
                    
                    # DATA TABLE
                    st.dataframe(results_df)