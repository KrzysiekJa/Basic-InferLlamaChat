# Basic-InferLlamaChat

The main focus of this project is on inference and a basic LLM web application serving approach, using an open-source model hosted on a third party hardware.

The project is currently in development and is not intended for production use.
The project is open source and can be found on [GitHub](https://github.com/KrzysiekJa/basic-inferllamachat).

## Setup instructions

1. Clone the repository: `git clone https://github.com/KrzysiekJa/basic-inferllamachat.git`
2. Navigate to the project directory: `cd basic-inferllamachat`
3. Install `uv` package manager, if not already installed: `pip install uv`
4. Create a virtual environment: `uv venv .venv`
5. Activate the virtual environment (`Linux/macOS`): `source .venv/bin/activate`
6. Install dependencies using command: `uv sync --locked --all-extras`
7. Create a API key for TogetherAI API on [https://together.xyz](https://together.xyz) and export it as an environment variable: `export TOGETHER_API_KEY=<your openai api key>`
8. Create a API key for OpenWeatherMap API on [https://openweathermap.org](https://openweathermap.org) and export it as an environment variable: `export OWM_API_KEY=<your owm api key>`, if you intend to use weather chatbot
9. Run the application: `PYTHONPATH=. python app/main.py`

\* For `Windows` users:

- activate the virtual environment using command: `.venv\Scripts\activate`,
- add `PYTHONPATH``variable following instructions from [this stackoverflow thread](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).
