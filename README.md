

# Retro Image Prompt Refiner
(readme generated by gemini 2.5 flash)
![Example Progress](main.png)

This project provides an interactive Gradio web application for iteratively refining image generation prompts using DSPy and the RetroDiffusion API.
Nothing serious, just proof of concept.

## Features

-   **Iterative Prompt Refinement:** Use DSPy to automatically revise your prompts based on the generated images.
-   **Optional User Feedback:** Provide ratings and comments at each step to guide the refinement process.
-   **Retro Diffusion Integration:** Generate images using the RetroDiffusion API with various styles, palettes, and input images.

## Prerequisites

-   Python 3.10+
-   Access to an LLM API (e.g., OpenAI, Anthropic, Gemini) with an API key.
-   Access to the Retro Diffusion API with an API token for using its features.

## Setup

1.  Clone this repository:

    ```bash
    git clone https://github.com/Archelunch/retro-refiner.git
    cd retro-refine
    ```

2.  Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Gradio application:

    ```bash
    python retro-dspy.py
    ```

4.  Open the provided local URL in your web browser.

## Usage example
Here's an example of how it works: https://x.com/mike_pavlukhin/status/1929249804904767984