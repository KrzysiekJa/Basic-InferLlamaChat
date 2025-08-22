import os

model_name = os.environ.get("MODEL_NAME", "gpt2")

if __name__ == "__main__":
    print(model_name)
