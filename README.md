# Basic-InferLlamaChat

The main focus of this project is on inference and a basic LLM web application serving approach, using an open-source model hosted on a third party hardware.

The project is currently in development and is not intended for production use.
The project is open source and can be found on [GitHub](https://github.com/KrzysiekJa/basic-inferllamachat).

## Demo video

https://github.com/user-attachments/assets/75d0434e-166e-4a69-97d8-10a98a00bc60

## Setup instructions

Whenever you intend to set up a project through your local virtual environment, follow these steps:

1. Clone the repository: `git clone https://github.com/KrzysiekJa/basic-inferllamachat.git`
2. Navigate to the project directory: `cd basic-inferllamachat`
3. Install the `uv` package manager, if not already installed: `pip install uv`
4. Create a virtual environment: `uv venv .venv`
5. Activate the virtual environment (`Linux/macOS`): `source .venv/bin/activate`
6. Install dependencies using the command: `uv sync --locked --all-extras`
7. Copy the `.env` file: `cp app/example.env app/.env`
8. Create an API key for the provider of your preference (OpenAI/TogetherAI/OpenRouter) and assign it within the `.env` file to the correct variable
9. Create an API key for the OpenWeatherMap API on [https://openweathermap.org](https://openweathermap.org) and place it in the `.env` file, if you intend to use the weather chatbot
10. Run the application: `PYTHONPATH=. python app/main.py`

\* For `Windows` users:

- activate the virtual environment using command: `.venv\Scripts\activate`,
- add `PYTHONPATH``variable following instructions from [this stackoverflow thread](https://stackoverflow.com/questions/3701646/how-to-add-to-the-pythonpath-in-windows-so-it-finds-my-modules-packages).
