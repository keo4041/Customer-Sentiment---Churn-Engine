Markdown# 🗣️ InsightFlow AI: Voice of Customer & Churn Intelligence

**InsightFlow AI** is a Streamlit-based intelligence engine that transforms raw customer support tickets into strategic product insights. It leverages multiple state-of-the-art LLMs (OpenAI, Gemini, Claude, Llama 3) to analyze sentiment, categorize topics, and predict customer churn risk in real-time.

---

## 🚀 Features

* **Multi-Model Intelligence:** "Bring Your Own Key" support for the world's top AI models:
    * **OpenAI:** GPT-4o
    * **Google:** Gemini 1.5 Flash
    * **Anthropic:** Claude 3.5 Sonnet
    * **Groq:** Llama 3 (70B)
* **Automated Sentiment Analysis:** Scores customer happiness on a scale of -1.0 to +1.0.
* **Churn Risk Prediction:** Identifies "High Risk" customers based on frustration signals and topic severity.
* **Strategic Insights:** Generates an executive summary for C-level decision-making.
* **Interactive Dashboard:** * Topic Distribution Charts (Altair)
    * Risk Matrix Scatter Plots
    * Metric scorecards (CSAT, High Risk Counts)
* **Mock Data Generator:** Built-in tool to generate realistic support ticket datasets for testing.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Data Manipulation:** Pandas
* **Visualization:** Altair
* **AI Providers:** `openai`, `google-generativeai`, `anthropic`, `groq`
* **Data Validation:** Pydantic

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/insightflow-ai.git](https://github.com/your-username/insightflow-ai.git)
cd insightflow-ai
2. Create a Virtual Environment (Optional but Recommended)Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install DependenciesBashpip install -r requirements.txt
4. Run the ApplicationBashstreamlit run app.py
🔑 Getting API KeysTo use this application, you will need an API key from one of the supported providers. You do not need all of them—just one is sufficient to run the analysis.ProviderModel UsedGet Key HereGoogleGemini 1.5 FlashGoogle AI StudioOpenAIGPT-4oOpenAI PlatformAnthropicClaude 3.5 SonnetAnthropic ConsoleGroqLlama 3 (70B)Groq Cloud📖 Usage GuideSelect Provider: On the sidebar, choose your preferred AI provider.Enter API Key: Paste your API Key into the password field (keys are not stored persistently).Load Data: Click the "🎲 Get Mock Data" button to generate a sample dataset of 20 customer tickets.Analyze: Click "🚀 Analyze Batch" on the main screen.Review Results:Metrics: See top-level CSAT and Churn counts.Insight: Read the AI-generated strategic advice.Graphs: Explore which topics are driving the most volume and risk.📂 Project Structureinsightflow-ai/
├── app.py                # Main application logic
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
🤝 ContributingContributions are welcome! Please feel free to submit a Pull Request.Fork the projectCreate your Feature Branch (git checkout -b feature/AmazingFeature)Commit your Changes (git commit -m 'Add some AmazingFeature')Push to the Branch (git push origin feature/AmazingFeature)Open a Pull Request📄 LicenseDistributed under the MIT License. See LICENSE for more information.
### Tips for your Repository
1.  **Add Screenshots:** After you run the app locally, take a screenshot of the dashboard and save it as `screenshot.png` in your folder. Then, add `![Dashboard Screenshot](screenshot.png)` right after the features section in the README. It makes the repo look much more professional.
2.  **Badges:** You can add badges (like "Built with Streamlit") to the top of the REA