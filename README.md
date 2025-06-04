# Healthcare Agentic AI System 🏥🤖

A comprehensive Agentic AI system for healthcare applications using Python and GPT-4. This system demonstrates how AI agents can automate tasks and make decisions in healthcare settings, enhancing efficiency and patient care through intelligent automation.

## 🌟 Features

### Multi-Agent Architecture
- **🗓️ Appointment Scheduling Agent**: Automated booking, rescheduling, and appointment management
- **💊 Drug Discovery Agent**: Compound analysis, treatment recommendations, and safety assessments  
- **📊 Patient Monitoring Agent**: Real-time vital signs analysis, risk assessment, and alert generation
- **🏥 General Healthcare Agent**: Medical consultations and clinical decision support

### Key Capabilities
- **Natural Language Processing**: Understands complex healthcare requests using GPT-4
- **Intelligent Routing**: Automatically routes requests to appropriate specialist agents
- **Risk Assessment**: Calculates patient risk scores and generates actionable insights
- **Alert System**: Multi-level alerts (Low, Medium, High, Critical) with recommended actions
- **Evidence-Based Recommendations**: Provides clinical guidelines-based suggestions

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key
- Internet connection

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/healthcare-agentic-ai.git
cd healthcare-agentic-ai
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up your OpenAI API key**
```bash
# Option 1: Environment variable (recommended)
export OPENAI_API_KEY="your-api-key-here"

# Option 2: Direct input when running tests
```

4. **Run the test suite**
```bash
python test_healthcare_ai.py
```

## 📁 Project Structure

```
healthcare-agentic-ai/
├── healthcare_ai.py          # Main AI system implementation
├── test_healthcare_ai.py     # Comprehensive testing suite
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── examples/                 # Usage examples and demos
│   ├── appointment_demo.py
│   ├── drug_discovery_demo.py
│   └── monitoring_demo.py
└── docs/                     # Documentation
    ├── API.md
    ├── ARCHITECTURE.md
    └── DEPLOYMENT.md
```

## 💻 Usage

### Basic Usage

```python
import asyncio
from healthcare_ai import HealthcareAI

# Initialize the system
ai_system = HealthcareAI("your-openai-api-key")

# Process healthcare requests
async def main():
    # Schedule an appointment
    result = await ai_system.process_natural_language_request(
        "Schedule an appointment for patient with diabetes next Tuesday"
    )
    print(result)
    
    # Monitor patient vitals
    monitoring_result = await ai_system.process_natural_language_request(
        "Monitor patient: heart rate 110, BP 150/95",
        context={"patient_id": "P001", "heart_rate": 110, "blood_pressure_systolic": 150}
    )
    print(monitoring_result)

asyncio.run(main())
```

### Testing Modes

The system includes three testing modes:

1. **Automated Full Test Suite**: Run all tests automatically
2. **Interactive Testing**: Choose specific agents to test
3. **Custom Request Testing**: Enter your own healthcare scenarios

## 🧪 Testing Examples

### Appointment Scheduling
```python
"Schedule an appointment for patient P001 with cardiology next Tuesday"
"Reschedule my appointment to next week due to emergency"
"What's the earliest available slot for diabetes consultation?"
```

### Drug Discovery
```python
"Analyze potential compounds for treating hypertension in elderly patients"
"What are safer alternatives to ACE inhibitors for kidney disease patients?"
"Provide safety analysis for compound XY-123 targeting cardiovascular disease"
```

### Patient Monitoring
```python
"Monitor patient: heart rate 105, blood pressure 160/95, temperature 100.2F"
"Assess cardiovascular risk for 65-year-old diabetic patient"
"Generate alerts for patient showing elevated glucose levels of 280 mg/dL"
```

### Complex Scenarios
```python
"Emergency: 68-year-old diabetic patient with chest pain, HR 115, BP 180/110"
"Optimize treatment for patient with muscle aches from statins and uncontrolled cholesterol"
```

## 📊 Sample Output

```json
{
  "action": "patient_monitoring", 
  "alerts": [
    {
      "patient_id": "P001",
      "alert_type": "Blood Pressure",
      "level": "HIGH",
      "message": "Hypertension detected - BP 160/95",
      "recommended_action": "Immediate medical evaluation required"
    }
  ],
  "status": "success"
}
```

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4  # Default model

# System Configuration
LOG_LEVEL=INFO
MAX_CONCURRENT_R
