# 💼 Investment Advisory AI Platform

[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Ftechiewonk%2Finvestadvisory%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/techiewonk/investadvisory/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/techiewonk/investadvisory)](https://github.com/techiewonk/investadvisory/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://github.com/techiewonk/investadvisory)

An AI-powered investment advisory service with intelligent agent architecture for financial analysis and portfolio management.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/), [FastAPI](https://fastapi.tiangolo.com/), and [Streamlit](https://streamlit.io/), this platform provides sophisticated financial analysis, market insights, and investment recommendations through advanced AI agents.

## 🚀 Features

### Core Capabilities
- **Financial Data Analysis**: Real-time market data integration with Yahoo Finance, Alpha Vantage, and FRED
- **Technical Analysis**: Advanced charting and technical indicators using TA-Lib
- **Portfolio Management**: Comprehensive portfolio analysis and optimization
- **Risk Assessment**: Multi-factor risk analysis and scenario modeling
- **Market Research**: AI-powered market research and trend analysis

### AI Agent Architecture
- **Research Assistant**: Web search and financial data analysis
- **RAG Assistant**: Knowledge base integration for financial documents
- **Portfolio Agent**: Specialized portfolio analysis and recommendations
- **Risk Agent**: Risk assessment and compliance checking
- **Market Agent**: Real-time market analysis and news processing

### Technical Features
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, Groq, and more
- **Advanced Streaming**: Real-time token and message streaming
- **Memory Persistence**: Conversation and knowledge persistence
- **Content Moderation**: LlamaGuard integration for compliance
- **Docker Support**: Complete containerization for easy deployment
- **Comprehensive Testing**: Unit and integration test coverage


## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional but recommended)
- At least one LLM API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/techiewonk/investadvisory.git
   cd investadvisory
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env to add your API keys
   echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
   ```

3. **Install dependencies:**
   ```bash
   # Using uv (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync --frozen
   
   # Or using pip
   pip install -e .
   ```

### Running the Application

#### Option 1: Docker (Recommended)
```bash
docker compose watch
```
- Streamlit app: http://localhost:8501
- API service: http://localhost:8080
- API docs: http://localhost:8080/redoc

#### Option 2: Local Development
```bash
# Terminal 1: Start the API service
python src/run_service.py

# Terminal 2: Start the Streamlit app
streamlit run src/streamlit_app.py
```

## 📊 Financial Data Sources

- **Yahoo Finance**: Real-time stock prices and historical data
- **Alpha Vantage**: Advanced market data and technical indicators
- **FRED**: Federal Reserve Economic Data
- **Pandas DataReader**: Multiple financial data sources
- **Custom APIs**: Extensible for additional data providers

## 🛠️ Development

### Project Structure
```
src/
├── agents/           # AI agent implementations
├── core/            # Core LLM and settings
├── schema/          # Data models and schemas
├── service/         # FastAPI service layer
├── client/          # HTTP client for API interaction
├── memory/          # Database and storage
└── streamlit_app.py # Web interface
```

### Adding New Agents
1. Create your agent in `src/agents/`
2. Add it to the agents registry in `src/agents/agents.py`
3. Update the Streamlit interface if needed

### Testing
```bash
# Install development dependencies
uv sync --frozen --group dev

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `GOOGLE_API_KEY`: Google API key
- `GROQ_API_KEY`: Groq API key
- `DATABASE_TYPE`: Database type (sqlite, postgres, mongo)
- `LANGCHAIN_API_KEY`: LangSmith tracing
- `LANGFUSE_PUBLIC_KEY`: Langfuse tracing

### Financial Data APIs
- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage API key
- `FRED_API_KEY`: FRED API key
- `OPENWEATHERMAP_API_KEY`: Weather data (optional)

## 📈 Use Cases

- **Individual Investors**: Personal portfolio analysis and recommendations
- **Financial Advisors**: Client portfolio management and research
- **Institutional Investors**: Large-scale portfolio optimization
- **Research Teams**: Market analysis and trend identification
- **Compliance Teams**: Risk assessment and regulatory compliance

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Hemprasad Badgujar**
- Email: hemprasad@badgujar.org
- GitHub: [@techiewonk](https://github.com/techiewonk)
- LinkedIn: [Hemprasad Badgujar](https://www.linkedin.com/in/hemprasad-badgujar/)

## 🙏 Acknowledgments

- Built on the foundation of the [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit)
- Powered by [LangGraph](https://langchain-ai.github.io/langgraph/) and [LangChain](https://langchain.com/)
- Financial data provided by Yahoo Finance, Alpha Vantage, and FRED

## 📞 Support

For support, please open an issue on GitHub or contact [hemprasad@badgujar.org](mailto:hemprasad@badgujar.org).

---

**⚠️ Disclaimer**: This software is for educational and research purposes only. It is not intended to provide financial advice. Always consult with qualified financial professionals before making investment decisions.