import gradio as gr

from fsi_core import (
    _app_cfg,
    analyze_stock,
)

# Multi-stock support: comma-separated symbols
def gradio_interface(symbols, start_date, end_date, investor_type):
    symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
    all_analyses = []
    all_recommendations = []
    all_plots = []
    all_inference_times = []
    all_token_counts = []
    all_data_points = []
    for symbol in symbol_list:
        result = analyze_stock(symbol, start_date, end_date, investor_type)
        all_analyses.append(f"[{symbol} - {investor_type} Investor]\n" + result["analysis"])
        all_recommendations.append(f"[{symbol}] {result['recommendation']}")
        all_plots.append(result["plot"])
        pm = result["performance_metrics"]
        all_inference_times.append(f"[{symbol}] LLM Inference Time: {pm.get('inference_time', 'N/A')}")
        all_token_counts.append(f"[{symbol}] Token Count: {pm.get('token_count', 'N/A')}")
        all_data_points.append(f"[{symbol}] Data Points: {pm.get('data_points', 'N/A')}")
    print("\n🕒 User-selected date range:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date:   {end_date}")
    print(f"   Investor Type: {investor_type}")
    print(f"   Symbols: {symbols}\n")
    # For plots, only show the first one if multiple
    return (
        '\n\n'.join(all_analyses),
        '\n'.join(all_recommendations),
        all_plots[0] if all_plots else None,
        '\n'.join(all_inference_times),
        '\n'.join(all_token_counts),
        '\n'.join(all_data_points)
    )

# Create an enhanced Gradio interface
def create_interface():
    # Custom CSS for Teal branding
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Arial:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Arial', Arial, sans-serif !important;
    }

    /* Teal accent color - PMS 3115 C */
    .primary {
        background: linear-gradient(135deg, #00C2DE 0%, #008AA8 100%) !important;
        border: none !important;
    }

    .primary:hover {
        background: linear-gradient(135deg, #008AA8 0%, #006A80 100%) !important;
    }

    /* Tab styling with Teal */
    .tab-nav button.selected {
        color: #00C2DE !important;
        border-bottom: 2px solid #00C2DE !important;
    }

    /* Headers with Teal accents */
    h1, h2, h3 {
        color: #2c3e50 !important;
    }

    /* Input focus states with Teal */
    input:focus, textarea:focus, select:focus {
        border-color: #00C2DE !important;
        box-shadow: 0 0 0 2px rgba(0, 194, 222, 0.1) !important;
    }

    /* Links and accents */
    a {
        color: #00C2DE !important;
    }

    /* Section headers */
    h3 {
        border-left: 4px solid #00C2DE !important;
        padding-left: 12px !important;
    }

    /* Increased container width to accommodate horizontal tabs */
    .gradio-container {
        max-width: 1600px !important;
        margin: auto !important;
        width: 100% !important;
    }

    /* Simplified tab styling - ensure visibility */
    .gradio-tabs {
        width: 100% !important;
        background: transparent !important;
    }

    /* Tab navigation styling */
    .gradio-tabs .tab-nav,
    .gradio-tabs > div:first-child {
        display: flex !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 8px !important;
        background: #f8f9fa !important;
        padding: 10px !important;
        border-radius: 10px !important;
        margin-bottom: 15px !important;
        width: 100% !important;
    }

    /* Individual tab buttons */
    .gradio-tabs .tab-nav button,
    .gradio-tabs > div:first-child > button {
        flex: 1 !important;
        min-width: 160px !important;
        max-width: 220px !important;
        padding: 10px 12px !important;
        border-radius: 6px !important;
        border: 2px solid #e0e0e0 !important;
        background: white !important;
        color: #666 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: block !important;
        visibility: visible !important;
    }

    .gradio-tabs .tab-nav button:hover {
        border-color: #00C2DE !important;
        color: #00C2DE !important;
        background: rgba(0, 194, 222, 0.05) !important;
    }

    .gradio-tabs .tab-nav button.selected {
        background: #00C2DE !important;
        color: white !important;
        border-color: #00C2DE !important;
    }

    /* Hide dropdown menu and force horizontal display */
    .gradio-tabs .tab-nav .tab-nav-button,
    .gradio-tabs button[aria-label="More tabs"],
    .gradio-tabs .tab-nav button:last-child[style*="display: none"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Force all tab buttons to be visible */
    .gradio-tabs .tab-nav button {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }

    /* Ensure all tabs are visible and container uses full width */
    .gradio-tabs .tab-nav {
        height: auto !important;
        max-height: none !important;
        width: 100% !important;
    }

    .gr-box {border-radius: 15px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);}

    /* Force tab navigation to be visible */
    .gradio-tabs .tab-nav,
    .gradio-tabs > div:first-child {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        background: #f8f9fa !important;
        padding: 12px !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
        gap: 8px !important;
        border: 1px solid #e0e0e0 !important;
        min-height: 50px !important;
    }
    .gradio-tabs .tab-nav button,
    .gradio-tabs > div:first-child > button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        flex: 1 !important;
        min-width: 150px !important;
        max-width: 200px !important;
        height: auto !important;
        padding: 8px 12px !important;
        background: white !important;
        border: 2px solid #e0e0e0 !important;
        border-radius: 6px !important;
        color: #333 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-align: center !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    .gradio-tabs .tab-nav button:hover,
    .gradio-tabs > div:first-child > button:hover {
        border-color: #00C2DE !important;
        background: rgba(0, 194, 222, 0.1) !important;
        color: #00C2DE !important;
    }
    .gradio-tabs .tab-nav button.selected,
    .gradio-tabs > div:first-child > button.selected,
    .gradio-tabs .tab-nav button[aria-selected="true"],
    .gradio-tabs > div:first-child > button[aria-selected="true"] {
        background: #00C2DE !important;
        border-color: #00C2DE !important;
        color: white !important;
    }
    .gradio-tabs button[aria-label*="More"],
    .gradio-tabs .tab-nav-button {
        display: none !important;
    }
    .gradio-tabs > div:nth-child(2) {
        background: transparent !important;
        border: none !important;
        margin-top: 10px !important;
    }
    """

    _js = """
    function ensureTabsVisible() {
        const tabContainers = document.querySelectorAll('.gradio-tabs');
        tabContainers.forEach(container => {
            const tabNav = container.querySelector('div:first-child');
            if (tabNav) {
                tabNav.style.display = 'flex';
                tabNav.style.flexWrap = 'nowrap';
                tabNav.style.gap = '8px';
                tabNav.style.width = '100%';
                const buttons = tabNav.querySelectorAll('button');
                buttons.forEach(btn => {
                    if (!btn.textContent.includes('...') && btn.textContent.trim() !== '') {
                        btn.style.display = 'block';
                        btn.style.visibility = 'visible';
                        btn.style.flex = '1';
                    } else {
                        btn.style.display = 'none';
                    }
                });
            }
        });
    }
    document.addEventListener('DOMContentLoaded', function() {
        setTimeout(ensureTabsVisible, 500);
        setTimeout(ensureTabsVisible, 1500);
    });
    """

    with gr.Blocks(title="AMD Instinct MI3xx ROCm-Powered Financial Analysis Demo") as interface:
        # Header with AMD logo in top right corner
        gr.HTML("""
            <div style="position: relative; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-bottom: 20px;">
                <img src="https://upload.wikimedia.org/wikipedia/commons/7/7c/AMD_Logo.svg" alt="AMD Logo" style="position: absolute; top: 15px; right: 20px; height: 35px; width: auto;" />
                <div style="padding-right: 120px;">
                    <h1 style="margin: 0; color: #2c3e50; font-size: 2.2em; font-weight: 700; font-family: Arial, sans-serif;"> AMD Instinct ROCm-Powered Financial Analysis Demo</h1>
                    <h3 style="margin: 5px 0 0 0; color: #00C2DE; font-size: 1.2em; font-weight: 600; font-family: Arial, sans-serif;">Powered by ROCm Platform </h3>
                </div>
            </div>
        """)

        gr.Markdown("""
            **Advanced AI-Driven Stock Analysis on AMD Hardware**

            This cutting-edge financial analysis tool leverages AMD's Instinct GPU architecture with ROCm platform
            to deliver high-performance AI-driven stock analysis with modern GPU acceleration.

            ### Key Features:
            - **AMD Instinct Architecture**: High HBM3 memory for complex financial modeling
            - **ROCm Software Stack**: Open-source GPU acceleration platform
            - **Real-time Processing**: GPU-accelerated technical indicators and market data
            - **Multi-Stock Portfolio Analysis**: Parallel processing capabilities

        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Analysis Configuration")

                symbols_input = gr.Textbox(
                    label="Stock Symbol(s) (e.g., AAPL, MSFT)",
                    placeholder="Enter one or more stock symbols, separated by commas...",
                    lines=2
                )

                # Date selection with calendar pickers
                gr.Markdown("#### 📅 Date Selection")
                with gr.Row():
                    with gr.Column():
                        start_date_input = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value=_app_cfg["default_start_date"],
                            placeholder="2024-01-01",
                            info="Enter start date in YYYY-MM-DD format"
                        )

                    with gr.Column():
                        end_date_input = gr.Textbox(
                            label="End Date (YYYY-MM-DD)",
                            value=_app_cfg["default_end_date"],
                            placeholder="2025-01-01",
                            info="Enter end date in YYYY-MM-DD format"
                        )

                investor_type_input = gr.Dropdown(
                    choices=["Conservative", "Moderate", "Aggressive", "Day Trader"],
                    label="Investor Type",
                    value="Moderate",
                    info="Select your investment profile for personalized recommendations"
                )

                analyze_btn = gr.Button("🚀 Analyze Stocks", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 📈 Analysis Results")

                with gr.Tabs():
                    with gr.TabItem("🤖 AI Technical Analysis"):
                        ai_analysis_output = gr.Textbox(
                            label="AI Technical & Market Analysis",
                            lines=15,
                            interactive=False
                        )

                    with gr.TabItem("💡 Buy/Sell/Hold Recommendations"):
                        recommendations_output = gr.Textbox(
                            label="Investment Recommendations",
                            lines=8,
                            interactive=False
                        )

                    with gr.TabItem("📊 Stock Charts & Price Analysis"):
                        chart_output = gr.Plot(label="Stock Price Chart (First Symbol)")

        # Performance Metrics moved down below main interface
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚡ System Performance Metrics")
                with gr.Row():
                    inference_time_output = gr.Textbox(label="LLM Inference Time (s)", interactive=False, scale=1)
                    token_count_output = gr.Textbox(label="Token Count", interactive=False, scale=1)
                    data_points_output = gr.Textbox(label="Data Points Analyzed", interactive=False, scale=1)


        # Event handlers
        analyze_btn.click(
            fn=gradio_interface,
            inputs=[symbols_input, start_date_input, end_date_input, investor_type_input],
            outputs=[
                ai_analysis_output,
                recommendations_output,
                chart_output,
                inference_time_output,
                token_count_output,
                data_points_output
            ]
        )

        # Example section
        gr.Markdown("""
            ### 💡 Example Usage
            1. Enter stock symbols (e.g., "AAPL, MSFT, GOOGL")
            2. Set your analysis date range
            3. Select your investor profile (Conservative, Moderate, Aggressive, Day Trader)
            4. Click "Analyze Stocks" to start GPU-accelerated analysis
            5. Review results across different analysis perspectives

            ### Disclaimer
            This tool is for educational purposes only. Always conduct your own research and consult with a financial advisor before making investment decisions.
        """)

    return interface, custom_css, _js

iface, _css, _js = create_interface()

if __name__ == "__main__":
    iface.launch(
        server_name=_app_cfg["host"],
        server_port=_app_cfg["port"],
        theme=gr.themes.Soft(),
        css=_css,
        js=_js,
    )
