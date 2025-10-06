
"""
Ultimate Timesheet Assistant - Complete Gradio Interface
Professional interface to interact with the Ultimate Expert Timesheet API
Supports all conversational features, project management, and timesheet operations
"""

import gradio as gr
import requests
import json
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import time
import asyncio

# Configuration
API_BASE_URL = "http://localhost:8000"
DEFAULT_EMAIL = "demo.user@company.com"

class UltimateTimesheetClient:
    """Professional client for Ultimate Timesheet API"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def chat(self, email: str, user_prompt: str) -> Dict:
        """Send chat message to API"""
        try:
            response = self.session.post(
                f"{self.base_url}/chat",
                json={"email": email, "user_prompt": user_prompt},
                timeout=30
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "response": f"❌ API Error: {response.status_code}\n{response.text}",
                    "conversation_phase": "error",
                    "tabular_data": None,
                    "suggestions": ["Try again", "Check API status"]
                }

        except requests.exceptions.ConnectionError:
            return {
                "response": "❌ Cannot connect to the Ultimate Timesheet API.\n\n"
                          "Please ensure the API server is running at http://localhost:8000\n\n"
                          "To start: python ultimate_expert_timesheet_api.py",
                "conversation_phase": "error",
                "tabular_data": None,
                "suggestions": ["Start API server", "Check connection"]
            }
        except Exception as e:
            return {
                "response": f"❌ Error: {str(e)}",
                "conversation_phase": "error",
                "tabular_data": None,
                "suggestions": ["Try again"]
            }

    def get_health(self) -> Dict:
        """Check API health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                status = health.get("status", "unknown")
                version = health.get("version", "unknown")
                expertise = health.get("expertise_level", "Expert")

                components = health.get("components", {})
                db_status = components.get("database", {}).get("status", "unknown")

                if status == "healthy":
                    return {
                        "status": "healthy",
                        "message": f"✅ Ultimate Timesheet API v{version} ({expertise})\n"
                                 f"🗄️ Database: {db_status}\n"
                                 f"🎯 All systems operational"
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "message": f"⚠️ API Status: {status}\n"
                                 f"Database: {db_status}\n"
                                 f"Version: {version}"
                    }
            else:
                return {
                    "status": "unhealthy",
                    "message": f"❌ HTTP {response.status_code}: {response.text}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"❌ Cannot reach Ultimate Timesheet API\n"
                         f"Error: {str(e)}\n"
                         f"URL: {self.base_url}"
            }

    def get_projects(self, system: Optional[str] = None) -> Dict:
        """Get project codes"""
        try:
            if system:
                response = self.session.get(f"{self.base_url}/projects/{system}")
            else:
                response = self.session.get(f"{self.base_url}/projects")

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "formatted_display": f"❌ Error getting projects: {response.status_code}",
                    "projects": [],
                    "count": 0
                }
        except Exception as e:
            return {
                "formatted_display": f"❌ Error: {str(e)}",
                "projects": [],
                "count": 0
            }

    def get_timesheet(self, email: str, system: str, start_date: str = None, end_date: str = None) -> Dict:
        """Get user timesheet"""
        try:
            params = {}
            if start_date:
                params['start_date'] = start_date
            if end_date:
                params['end_date'] = end_date

            response = self.session.get(
                f"{self.base_url}/timesheet/{email}/{system}",
                params=params
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "formatted_display": f"❌ Error getting timesheet: {response.status_code}",
                    "entries": [],
                    "total_hours": 0,
                    "count": 0
                }
        except Exception as e:
            return {
                "formatted_display": f"❌ Error: {str(e)}",
                "entries": [],
                "total_hours": 0,
                "count": 0
            }

# Initialize API client
api_client = UltimateTimesheetClient()

# Global conversation history for the interface
conversation_history = []

def add_to_history(user_msg: str, bot_response: str, phase: str = ""):
    """Add exchange to conversation history"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    conversation_history.append({
        "timestamp": timestamp,
        "user": user_msg,
        "bot": bot_response,
        "phase": phase
    })

def format_conversation_display() -> str:
    """Format conversation for display"""
    if not conversation_history:
        return "💬 **Conversation will appear here**\n\nStart by typing a message or try one of the examples!"

    formatted = []
    for entry in conversation_history[-8:]:  # Show last 8 exchanges
        formatted.append(f"**[{entry['timestamp']}] 👤 You:**")
        formatted.append(f"{entry['user']}\n")
        formatted.append(f"**🤖 Assistant:** ({entry.get('phase', 'unknown')})")
        formatted.append(f"{entry['bot']}\n")
        formatted.append("---\n")

    return "\n".join(formatted)

# Main chat processing function
def process_chat(email: str, message: str, history):
    """Process chat message through API"""
    if not email.strip():
        error_msg = "⚠️ Please enter your email address first"
        history.append([message, error_msg])
        add_to_history(message, error_msg, "error")
        return history, "", format_conversation_display()

    if not message.strip():
        return history, "", format_conversation_display()

    # Call the Ultimate API
    result = api_client.chat(email.strip(), message.strip())

    # Extract response components
    bot_response = result.get("response", "No response received")
    conversation_phase = result.get("conversation_phase", "unknown")
    tabular_data = result.get("tabular_data")
    suggestions = result.get("suggestions", [])

    # Enhance response with tabular data
    if tabular_data:
        enhanced_response = f"{bot_response}\n\n{tabular_data}"
    else:
        enhanced_response = bot_response

    # Add suggestions if available
    if suggestions:
        enhanced_response += "\n\n💡 **Suggestions:**\n"
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            enhanced_response += f"• {suggestion}\n"

    # Update conversation history
    history.append([message, enhanced_response])
    add_to_history(message, enhanced_response, conversation_phase)

    return history, "", format_conversation_display()

def clear_conversation():
    """Clear conversation history"""
    global conversation_history
    conversation_history = []
    return [], "", "💬 **Conversation cleared!**\n\nStart fresh with your timesheet questions."

def check_api_status():
    """Check API status"""
    health = api_client.get_health()
    return health.get("message", "Unable to check API status")

def show_projects(system):
    """Show project codes for system"""
    if not system:
        return "⚠️ Please select a system (Oracle or Mars)"

    result = api_client.get_projects(system)

    # Create DataFrame for download
    projects_df = None
    if result.get("projects"):
        projects_df = pd.DataFrame(result["projects"])

    return result.get("formatted_display", "No projects found"), projects_df

def show_timesheet(email, system, start_date, end_date):
    """Show user timesheet"""
    if not email.strip():
        return "⚠️ Please enter your email address", None

    if not system:
        return "⚠️ Please select a system (Oracle or Mars)", None

    # Format dates
    start_str = start_date.strftime("%Y-%m-%d") if start_date else None
    end_str = end_date.strftime("%Y-%m-%d") if end_date else None

    result = api_client.get_timesheet(email, system, start_str, end_str)

    # Create DataFrame for download
    timesheet_df = None
    if result.get("entries"):
        timesheet_df = pd.DataFrame(result["entries"])

    return result.get("formatted_display", "No timesheet entries found"), timesheet_df

def submit_quick_entry(email, system, date_input, hours, project_code, task_code, comments):
    """Submit a quick timesheet entry"""
    if not all([email.strip(), system, date_input, hours, project_code.strip()]):
        return "⚠️ Please fill all required fields (Email, System, Date, Hours, Project Code)"

    # Format the entry as a conversational prompt
    date_str = date_input.strftime("%Y-%m-%d") if hasattr(date_input, 'strftime') else str(date_input)

    prompt_parts = [f"{hours} hours", f"{system} project {project_code}", f"on {date_str}"]

    if task_code.strip():
        prompt_parts.append(f"task {task_code}")

    if comments.strip():
        prompt_parts.append(f"comments: {comments}")

    conversation_prompt = ", ".join(prompt_parts)

    # Process through chat API to maintain conversation flow
    result = api_client.chat(email, conversation_prompt)

    return result.get("response", "Entry processing failed")

# Custom CSS for beautiful styling
custom_css = """
/* Ultimate Timesheet App Styling */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

.main-header {
    text-align: center;
    color: white;
    padding: 30px 20px;
    margin-bottom: 30px;
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.main-header h1 {
    font-size: 2.8em;
    margin: 0 0 15px 0;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    font-weight: bold;
}

.main-header p {
    font-size: 1.2em;
    margin: 0;
    opacity: 0.95;
}

.card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.status-display {
    font-family: 'Courier New', monospace;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    white-space: pre-line;
}

.conversation-display {
    max-height: 400px;
    overflow-y: auto;
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    white-space: pre-line;
    font-size: 14px;
    line-height: 1.5;
}

.quick-entry-form {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin: 15px 0;
}

.quick-entry-form .form-group {
    margin-bottom: 15px;
}

.examples-section {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    margin: 20px 0;
}

.examples-section h3 {
    margin-top: 0;
    color: white;
}

.examples-list {
    list-style: none;
    padding: 0;
    margin: 15px 0;
}

.examples-list li {
    background: rgba(255, 255, 255, 0.2);
    padding: 8px 12px;
    margin: 5px 0;
    border-radius: 8px;
    cursor: pointer;
    transition: background 0.3s;
}

.examples-list li:hover {
    background: rgba(255, 255, 255, 0.3);
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin: 20px 0;
}

.feature-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
}

.tab-content {
    padding: 20px;
    background: white;
    border-radius: 10px;
    margin-top: 10px;
}
"""

def create_ultimate_interface():
    """Create the ultimate Gradio interface"""

    with gr.Blocks(
        css=custom_css,
        title="🎯 Ultimate Timesheet Assistant - 50+ Years Expertise",
        theme=gr.themes.Soft()
    ) as demo:

        # Header
        gr.HTML("""
        <div class='main-header'>
            <h1>🎯 Ultimate Timesheet Assistant</h1>
            <p>Professional conversational timesheet management with 50+ years of expertise</p>
            <p><strong>Oracle & Mars Systems | Multi-Entry Support | Expert AI Guidance</strong></p>
        </div>
        """)

        with gr.Tabs() as tabs:

            # Tab 1: Conversational Chat
            with gr.TabItem("💬 Conversational Assistant", elem_classes=["tab-content"]):
                gr.HTML("<h2>🎯 Natural Language Timesheet Management</h2>")

                with gr.Row():
                    with gr.Column(scale=1):
                        email_input = gr.Textbox(
                            label="📧 Your Email Address",
                            value=DEFAULT_EMAIL,
                            placeholder="Enter your company email",
                            lines=1
                        )

                        # Main chat interface
                        chatbot = gr.Chatbot(
                            label="💬 Expert Conversation",
                            height=500,
                            placeholder="Your conversation with the expert assistant will appear here..."
                        )

                        with gr.Row():
                            message_input = gr.Textbox(
                                label="💭 Your Message",
                                placeholder="Type your timesheet request... (e.g., '8 hours Oracle ORG-001 yesterday')",
                                lines=2,
                                scale=4
                            )

                        with gr.Row():
                            send_btn = gr.Button("📤 Send", variant="primary", scale=1)
                            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", scale=1)

                    with gr.Column(scale=1):
                        # API Status
                        api_status = gr.Textbox(
                            label="🔧 API Status",
                            value="Click 'Check Status' to test connection",
                            interactive=False,
                            lines=4,
                            elem_classes=["status-display"]
                        )

                        check_status_btn = gr.Button("🔍 Check API Status", variant="secondary")

                        # Conversation Timeline
                        conversation_display = gr.Textbox(
                            label="📋 Conversation Timeline",
                            value="💬 **Conversation will appear here**\n\nStart by typing a message!",
                            interactive=False,
                            lines=8,
                            elem_classes=["conversation-display"]
                        )

                # Examples section
                gr.HTML("""
                <div class='examples-section'>
                    <h3>🎯 Example Commands & Natural Language</h3>
                    <div class='feature-grid'>
                        <div class='feature-card'>
                            <h4>📝 Add Entries</h4>
                            <ul class='examples-list'>
                                <li>"8 hours Oracle ORG-001 yesterday"</li>
                                <li>"Mars: 4 hours MRS-002, Oracle: 4 hours ORG-003, both today"</li>
                                <li>"6 hours ORG-001 today, task DEV-001, database work"</li>
                            </ul>
                        </div>
                        <div class='feature-card'>
                            <h4>📊 View Data</h4>
                            <ul class='examples-list'>
                                <li>"show my Oracle timesheet"</li>
                                <li>"show timesheet Mars"</li>
                                <li>"show my entries from last week"</li>
                            </ul>
                        </div>
                        <div class='feature-card'>
                            <h4>📋 Get Help</h4>
                            <ul class='examples-list'>
                                <li>"show projects Oracle"</li>
                                <li>"help"</li>
                                <li>"start fresh"</li>
                            </ul>
                        </div>
                    </div>
                </div>
                """)

            # Tab 2: Project Codes
            with gr.TabItem("📋 Project Codes", elem_classes=["tab-content"]):
                gr.HTML("<h2>📋 Available Project Codes</h2>")

                with gr.Row():
                    system_selector = gr.Dropdown(
                        choices=["Oracle", "Mars"],
                        label="🔧 Select System",
                        value="Oracle"
                    )
                    get_projects_btn = gr.Button("📋 Get Project Codes", variant="primary")

                with gr.Row():
                    projects_display = gr.Textbox(
                        label="📊 Project Codes",
                        interactive=False,
                        lines=15,
                        elem_classes=["status-display"]
                    )

                with gr.Row():
                    projects_download = gr.File(
                        label="📥 Download Project Codes CSV",
                        interactive=False
                    )

            # Tab 3: Timesheet Viewer
            with gr.TabItem("📊 Timesheet Viewer", elem_classes=["tab-content"]):
                gr.HTML("<h2>📊 View Your Timesheet Entries</h2>")

                with gr.Row():
                    with gr.Column():
                        viewer_email = gr.Textbox(
                            label="📧 Email",
                            value=DEFAULT_EMAIL,
                            placeholder="Enter email address"
                        )

                        viewer_system = gr.Dropdown(
                            choices=["Oracle", "Mars"],
                            label="🔧 System",
                            value="Oracle"
                        )

                    with gr.Column():
                        start_date = gr.DateTime(
                            label="📅 Start Date (optional)",
                            include_time=False
                        )

                        end_date = gr.DateTime(
                            label="📅 End Date (optional)",
                            include_time=False
                        )

                with gr.Row():
                    get_timesheet_btn = gr.Button("📊 Get Timesheet", variant="primary")

                with gr.Row():
                    timesheet_display = gr.Textbox(
                        label="📋 Your Timesheet",
                        interactive=False,
                        lines=15,
                        elem_classes=["status-display"]
                    )

                with gr.Row():
                    timesheet_download = gr.File(
                        label="📥 Download Timesheet CSV",
                        interactive=False
                    )

            # Tab 4: Quick Entry Form
            with gr.TabItem("⚡ Quick Entry", elem_classes=["tab-content"]):
                gr.HTML("""
                <div class='quick-entry-form'>
                    <h2>⚡ Quick Timesheet Entry Form</h2>
                    <p>Fill out a single timesheet entry using form fields</p>
                </div>
                """)

                with gr.Row():
                    with gr.Column():
                        quick_email = gr.Textbox(
                            label="📧 Email",
                            value=DEFAULT_EMAIL
                        )

                        quick_system = gr.Dropdown(
                            choices=["Oracle", "Mars"],
                            label="🔧 System",
                            value="Oracle"
                        )

                        quick_date = gr.DateTime(
                            label="📅 Date",
                            include_time=False,
                            value=datetime.now().date()
                        )

                    with gr.Column():
                        quick_hours = gr.Number(
                            label="⏰ Hours",
                            value=8.0,
                            minimum=0.25,
                            maximum=24.0,
                            step=0.25
                        )

                        quick_project = gr.Textbox(
                            label="📂 Project Code",
                            placeholder="e.g., ORG-001, MRS-002"
                        )

                        quick_task = gr.Textbox(
                            label="📋 Task Code (optional)",
                            placeholder="e.g., DEV-001"
                        )

                with gr.Row():
                    quick_comments = gr.Textbox(
                        label="💬 Comments (optional)",
                        placeholder="Describe your work...",
                        lines=3
                    )

                with gr.Row():
                    submit_quick_btn = gr.Button("✅ Submit Entry", variant="primary", size="lg")

                with gr.Row():
                    quick_result = gr.Textbox(
                        label="📋 Result",
                        interactive=False,
                        lines=5,
                        elem_classes=["status-display"]
                    )

            # Tab 5: Help & Documentation
            with gr.TabItem("❓ Help & Documentation", elem_classes=["tab-content"]):
                gr.HTML("""
                <div class='card'>
                    <h2>🎯 Ultimate Timesheet Assistant - Help</h2>

                    <h3>🚀 Getting Started</h3>
                    <p><strong>The Ultimate Timesheet Assistant</strong> uses natural language to help you manage your Oracle and Mars timesheets with 50+ years of professional expertise.</p>

                    <h3>💬 Conversational Features</h3>
                    <ul>
                        <li><strong>Natural Language:</strong> "8 hours Oracle ORG-001 yesterday"</li>
                        <li><strong>Multi-System Support:</strong> "Oracle: 4 hours ORG-001, Mars: 4 hours MRS-002, both today"</li>
                        <li><strong>Intelligent Prompting:</strong> The AI asks for missing information</li>
                        <li><strong>Confirmation Flow:</strong> Always confirms before submitting</li>
                    </ul>

                    <h3>📋 Available Commands</h3>
                    <table style="width:100%; border-collapse: collapse; margin: 20px 0;">
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 12px; border: 1px solid #ddd;">Command</th>
                            <th style="padding: 12px; border: 1px solid #ddd;">Description</th>
                            <th style="padding: 12px; border: 1px solid #ddd;">Example</th>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><code>show projects [system]</code></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">Display project codes</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">"show projects Oracle"</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><code>show timesheet [system]</code></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">View your entries</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">"show timesheet Mars"</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><code>help</code></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">Get assistance</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">"help"</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><code>start fresh</code></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">Clear session</td>
                            <td style="padding: 8px; border: 1px solid #ddd;">"start fresh"</td>
                        </tr>
                    </table>

                    <h3>🎯 Expert Features</h3>
                    <ul>
                        <li><strong>Anti-Hallucination:</strong> Only uses information you provide</li>
                        <li><strong>Multi-System Booking:</strong> Oracle and Mars in one conversation</li>
                        <li><strong>Draft Management:</strong> Save before final submission</li>
                        <li><strong>Professional Validation:</strong> Ensures data accuracy</li>
                        <li><strong>Tabular Display:</strong> Beautiful formatting without HTML</li>
                    </ul>

                    <h3>🔧 Setup Requirements</h3>
                    <ol>
                        <li>Start the Ultimate Timesheet API server</li>
                        <li>Ensure database connection is working</li>
                        <li>Configure your email address</li>
                        <li>Begin conversational timesheet management!</li>
                    </ol>

                    <h3>📞 API Endpoints</h3>
                    <ul>
                        <li><strong>POST /chat</strong> - Main conversational interface</li>
                        <li><strong>GET /projects/{system}</strong> - Project codes</li>
                        <li><strong>GET /timesheet/{email}/{system}</strong> - Timesheet entries</li>
                        <li><strong>GET /health</strong> - API health check</li>
                    </ul>
                </div>
                """)

        # Event Handlers

        # Chat interface events
        send_btn.click(
            fn=process_chat,
            inputs=[email_input, message_input, chatbot],
            outputs=[chatbot, message_input, conversation_display]
        )

        message_input.submit(
            fn=process_chat,
            inputs=[email_input, message_input, chatbot],
            outputs=[chatbot, message_input, conversation_display]
        )

        clear_btn.click(
            fn=clear_conversation,
            outputs=[chatbot, message_input, conversation_display]
        )

        # API status check
        check_status_btn.click(
            fn=check_api_status,
            outputs=[api_status]
        )

        # Project codes events
        get_projects_btn.click(
            fn=show_projects,
            inputs=[system_selector],
            outputs=[projects_display, projects_download]
        )

        # Timesheet viewer events
        get_timesheet_btn.click(
            fn=show_timesheet,
            inputs=[viewer_email, viewer_system, start_date, end_date],
            outputs=[timesheet_display, timesheet_download]
        )

        # Quick entry events
        submit_quick_btn.click(
            fn=submit_quick_entry,
            inputs=[quick_email, quick_system, quick_date, quick_hours, quick_project, quick_task, quick_comments],
            outputs=[quick_result]
        )

        # Auto-check API status on load
        demo.load(
            fn=check_api_status,
            outputs=[api_status]
        )

    return demo

if __name__ == "__main__":
    print("🚀 Starting Ultimate Timesheet Assistant Interface...")
    print("🎯 50+ Years of Professional Expertise")
    print(f"🌐 API URL: {API_BASE_URL}")
    print("📧 Default Email:", DEFAULT_EMAIL)
    print("💡 Make sure the Ultimate Timesheet API is running!")
    print("🔗 Interface will be available at: http://localhost:7860")

    demo = create_ultimate_interface()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
        show_error=True,
        show_api=False
    )
