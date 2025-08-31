import os
import json
from typing import Any, Generator
from httpx import Client
from openai import OpenAI
from pydantic import BaseModel, Field

TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", None)
OWM_API_KEY = os.environ.get("OWM_API_KEY", None)  # openweathermap api key
client = OpenAI(api_key=TOGETHER_API_KEY, base_url="https://api.together.xyz")


def run_batch_inference(
    user_prompt: str, system_instruction: str = "You are a helpful AI assistant."
) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
    )
    return chat_completion.choices[0].message.content


def run_stream_inference(
    user_prompt: str, system_instruction: str = "You are a helpful AI assistant."
) -> Generator[Any, Any, Any]:
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
        stream=True,
    )

    for chunk in stream:
        try:
            content = chunk.choices[0].delta.content or ""
            yield content
        except IndexError:
            # Handle the case where chunk.choices is empty
            yield ""


class User(BaseModel):
    name: str = Field(..., description="The name of the user", example="John Doe")
    address: str = Field(
        ..., description="The address of the user", example="123 Main St"
    )


def get_inference_schema(
    user_prompt: str, system_instruction: str = "You are a helpful AI assistant."
) -> User:
    chat_completion = client.chat.completions.create(
        response_format={"type": "json_object", "schema": User.model_json_schema()},
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt},
        ],
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=128,
    )

    create_user = json.loads(chat_completion.choices[0].message.content)
    return create_user


def get_current_weather(location: str, unit: str = "celsius") -> str:
    if "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": 70})
    elif "london" in location.lower():
        return json.dumps({"location": "London", "temperature": 60})
    elif "new york" in location.lower():
        return json.dumps({"location": "New York", "temperature": 75})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


def get_current_weather_from_owm(
    location: str = "New York, US", unit: str = "metric"
) -> str:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather?"
    url = f"{BASE_URL}q={location}&appid={OWM_API_KEY}&units={unit}"

    with Client() as client:
        response = client.get(url, timeout=10)

    if response.status_code == 200:
        data = response.json()
        return json.dumps(
            {
                "location": data["name"],
                "country": data["sys"]["country"],
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "pressure": data["main"]["pressure"],
                "feels_like": data["main"]["feels_like"],
                "wind_speed": data["wind"]["speed"],
                "description": data["weather"][0]["description"].capitalize(),
                "temperature_unit": "°C"
                if unit == "metric"
                else "°F"
                if unit == "imperial"
                else "K",
                "speed_unit": "m/s" if unit == "metric" else "mph",
            }
        )
    return json.dumps(
        {"location": location, "temperature": "unknown", "description": "unknown"}
    )


def run_inference_tool(user_prompt: str) -> Any:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather_from_owm",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["metric", "imperial"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant that can access external functions. The responses from these functions will be appended to this dialog. Please, provide responses based on the information from these function calls.",
        },
        {"role": "user", "content": user_prompt},
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=128,
        tools=tools,
        tool_choice="auto",
    )

    tool_calls = chat_completion.choices[0].message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            if function_name == "get_current_weather_from_owm":
                function_response = get_current_weather_from_owm(
                    function_args.get("location"), function_args.get("unit", "metric")
                )
                messages.append(
                    {
                        "toll_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )

    function_enriched_response = client.chat.completions.create(
        messages=messages,
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        max_tokens=512,
    )
    return function_enriched_response.choices[0].message


def main():
    result = run_batch_inference("Tell about Paris.")
    print(result)

    for part in run_stream_inference("Tell me about Madrid and Warsaw."):
        print(part, end="", flush=True)

    print("\n")

    result = get_inference_schema(
        user_prompt="Create a user named Jane Doe, who lives at 97, Smith Street."
    )
    print(result)

    print(get_current_weather_from_owm("New York"))

    result = run_inference_tool(
        "What is the temperature in New York and Paris and London"
    )
    print(json.dumps(result.model_dump(), indent=4), "\n")
    print(json.dumps(result.model_dump()["content"]))


if __name__ == "__main__":
    main()
