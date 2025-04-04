# Subject Analyzer

A modular Python application that analyzes text to identify and classify its main subject using the DeepSeek R1 language model.

## Features

- Identifies main subject from text input
- Classifies subject type (Person, Place, Concept, etc.)
- Extracts key aspects of the subject
- Determines the domain/field
- Provides confidence scoring
- Rich terminal interface with progress indicators

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd subject_analyzer
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file with:
```
DEEPSEEK_API_KEY=your_api_key
DEEPSEEK_BASE_URL=https://api.deepinfra.com/v1/openai
TEMPERATURE=0.7
MODEL_NAME=deepseek-ai/DeepSeek-R1
MAX_RETRIES=3
TIMEOUT=30
```

## Usage

Run the application:
```bash
python main.py
```

Enter text when prompted, and the analyzer will:
1. Identify the main subject
2. Classify its type
3. Extract key aspects
4. Determine the domain
5. Provide a confidence score

## Architecture

The application follows SOLID principles with a modular design:

- `models/`: Data classes and type definitions
- `interfaces/`: Abstract base classes defining contracts
- `services/`: Core business logic implementations
- `utils/`: Helper utilities and tools
- `config/`: Configuration management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 