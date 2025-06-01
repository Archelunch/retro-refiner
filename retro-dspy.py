from PIL import Image
from io import BytesIO
import requests
import base64
from typing import List, Tuple, Generator, Dict, Any, Optional
from functools import partial

import dspy
import gradio as gr

API_KEY = None
CURRENT_MODEL = None
LLM_INSTANCE = None
RETRO_DIFFUSION_TOKEN = None

def configure_llm(api_key: str, model_name: str) -> None:
    global API_KEY, CURRENT_MODEL, LLM_INSTANCE
    API_KEY = api_key
    CURRENT_MODEL = model_name
    LLM_INSTANCE = dspy.LM(
        model_name,
        api_key=api_key,
        max_tokens=20384,
        temperature=1.0,
    )
    dspy.configure(lm=LLM_INSTANCE)

def configure_retro_diffusion(token: str) -> None:
    global RETRO_DIFFUSION_TOKEN
    RETRO_DIFFUSION_TOKEN = token

# --- Core Image and DSPy Functions (No UI Dependencies) ---


def get_base64_from_path(input_image_path: str) -> str:
    with Image.open(input_image_path) as img:
        rgb_img = img.convert("RGB")
        buffer = BytesIO()
        rgb_img.save(buffer, format="PNG")
        base64_input_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_input_image


def get_base64_from_pil(pil_image: Image.Image) -> str:
    rgb_img = pil_image.convert("RGB")
    buffer = BytesIO()
    rgb_img.save(buffer, format="PNG")
    base64_input_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_input_image


def from_base64_image(base64_image: str) -> tuple[dspy.Image, Image.Image]:
    image_data = base64.b64decode(base64_image)
    pil_image = Image.open(BytesIO(image_data))
    try:
        dspy_image = dspy.Image.from_PIL(pil_image)
    except AttributeError:
        print(
            "Warning: dspy.Image.from_PIL might not be the correct method. Check DSPy documentation if image processing fails."
        )
        dspy_image = pil_image  # Placeholder
    return dspy_image, pil_image


def get_retro_diffusion_image(
    prompt: str,
    input_palette_pil: Image.Image | None = None,
    input_image_pil: Image.Image | None = None,
    strength: float = 0.8,
    width: int = 150,
    height: int = 150,
    prompt_style: str = "rd_plus__retro",
    seed: int | None = None,
) -> tuple[dspy.Image, Image.Image]:
    base64_palette_image = None
    if input_palette_pil:
        base64_palette_image = get_base64_from_pil(input_palette_pil)

    base64_input_image = None
    if input_image_pil:
        base64_input_image = get_base64_from_pil(input_image_pil)

    url = "https://api.retrodiffusion.ai/v1/inferences"
    method = "POST"
    headers = {
        "X-RD-Token": RETRO_DIFFUSION_TOKEN
    }
    payload = {
        "width": width,
        "height": height,
        "prompt": prompt,
        "num_images": 1,
        "prompt_style": prompt_style,
    }
    if base64_palette_image:
        payload["input_palette"] = base64_palette_image
    if base64_input_image:
        payload["input_image"] = base64_input_image
        payload["strength"] = strength
    if seed is not None and seed > 0:
        payload["seed"] = seed
    response = requests.request(method, url, headers=headers, json=payload)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    return from_base64_image(response.json()["base64_images"][0])


def save_image(
    image: Image.Image, output_path: str
):  # Not directly used by Gradio app but kept
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)


# --- DSPy Predictor Setup ---
_check_and_revise_prompt_predictor_standard: Optional[dspy.Predict] = None
_check_and_revise_prompt_predictor_with_feedback: Optional[dspy.Predict] = None

DEFAULT_DSPY_SIG_STANDARD = "desired_prompt: str, current_image: dspy.Image, current_prompt:str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str"
DEFAULT_DSPY_SIG_WITH_FEEDBACK = "desired_prompt: str, current_image: dspy.Image, current_prompt:str, user_rating: int, user_comment: str -> feedback:str, image_strictly_matches_desired_prompt: bool, revised_prompt: str"


def get_dspy_predictor(with_feedback: bool = False) -> dspy.Predict:
    global \
        _check_and_revise_prompt_predictor_standard, \
        _check_and_revise_prompt_predictor_with_feedback
    if with_feedback:
        if _check_and_revise_prompt_predictor_with_feedback is None:
            _check_and_revise_prompt_predictor_with_feedback = dspy.Predict(
                DEFAULT_DSPY_SIG_WITH_FEEDBACK
            )
        return _check_and_revise_prompt_predictor_with_feedback
    else:
        if _check_and_revise_prompt_predictor_standard is None:
            _check_and_revise_prompt_predictor_standard = dspy.Predict(
                DEFAULT_DSPY_SIG_STANDARD
            )
        return _check_and_revise_prompt_predictor_standard


# --- Helper Functions for UI and State Management (Ordered before main generators) ---
def _create_ui_update_tuple(
    status: Dict[str, Any], new_log_messages: Optional[List[str]] = None
) -> Tuple[Any, ...]:
    if new_log_messages:
        string_log_messages = [str(msg) for msg in new_log_messages]
        status.get("accumulated_log_entries", []).extend(string_log_messages)

    is_processing = status.get("is_processing_something", False)
    is_awaiting = status.get("is_awaiting_feedback", False)
    current_iter = status.get("current_iter_num", 0)
    max_iter_val = status.get("max_iter", 0)
    user_stopped = status.get("user_forced_stop", False)
    is_completed = user_stopped or (current_iter >= max_iter_val and current_iter > 0)

    main_btn_interactive = True
    main_btn_value = "🚀 Generate & Refine Image"
    main_inputs_interactive = True

    if is_processing:
        main_btn_value = f"⏳ Processing Iter {current_iter}/{max_iter_val}..."
        main_btn_interactive = False
        main_inputs_interactive = False
    elif is_awaiting:
        main_btn_value = "⏳ Waiting for Your Feedback..."
        main_btn_interactive = False
        main_inputs_interactive = False
    elif is_completed:
        main_btn_value = "✅ Done! Run Again?"
    # Else: Idle, ready for a new run, defaults are fine

    feedback_accordion_label = f"✍️ Provide Your Feedback for Iteration {current_iter}"
    feedback_accordion_visible = status.get("feedback_enabled", False) and is_awaiting
    feedback_buttons_interactive = feedback_accordion_visible

    return (
        status.get("current_pil_image"),
        status.get("current_prompt", ""),
        "\n".join(status.get("accumulated_log_entries", [])),
        status.get("accumulated_gallery_items", []),
        gr.update(value=main_btn_value, interactive=main_btn_interactive),
        gr.update(interactive=main_inputs_interactive),  # initial_prompt_input
        gr.update(interactive=main_inputs_interactive),  # max_iter_slider
        gr.update(interactive=main_inputs_interactive),  # palette_image_input
        gr.update(interactive=main_inputs_interactive),  # seed_input_text
        gr.update(interactive=main_inputs_interactive),  # feedback_enabled_checkbox
        gr.update(visible=feedback_accordion_visible, label=feedback_accordion_label),
        gr.update(interactive=feedback_buttons_interactive, value=3),
        gr.update(interactive=feedback_buttons_interactive, value=""),
        gr.update(interactive=feedback_buttons_interactive),
        gr.update(interactive=feedback_buttons_interactive),
        gr.update(interactive=feedback_buttons_interactive),
        status.copy(),
    )


def _prepare_initial_state(
    initial_prompt_ui: str,
    max_iter_ui: int,
    palette_image_ui: Optional[Image.Image],
    input_image_ui: Optional[Image.Image],
    strength_ui: float,
    width_ui: int,
    height_ui: int,
    prompt_style_ui: str,
    seed_str_ui: str,
    feedback_enabled_ui: bool,
) -> Dict[str, Any]:
    new_status: Dict[str, Any] = {}
    new_status["initial_prompt"] = initial_prompt_ui
    new_status["max_iter"] = int(max_iter_ui)
    new_status["input_palette_pil"] = palette_image_ui
    new_status["input_image_pil"] = input_image_ui
    new_status["strength"] = float(strength_ui)
    new_status["width"] = int(width_ui)
    new_status["height"] = int(height_ui)
    new_status["prompt_style"] = prompt_style_ui
    new_status["feedback_enabled"] = feedback_enabled_ui
    parsed_seed = None
    if seed_str_ui and seed_str_ui.strip():
        try:
            parsed_seed = int(seed_str_ui)
        except ValueError:
            parsed_seed = None
        if parsed_seed is not None and parsed_seed <= 0:
            parsed_seed = None
    new_status["seed"] = parsed_seed
    new_status["current_iter_num"] = 0
    new_status["current_prompt"] = initial_prompt_ui
    new_status["current_dspy_image"] = None
    new_status["current_pil_image"] = None
    new_status["is_awaiting_feedback"] = False
    new_status["user_forced_stop"] = False
    new_status["accumulated_gallery_items"] = []
    new_status["accumulated_log_entries"] = []
    new_status["last_dspy_result"] = None
    new_status["is_processing_something"] = False
    return new_status


# --- Main Generator Functions for the Interactive Process ---


def handle_user_feedback_and_proceed(
    app_status: Dict[str, Any],
    user_rating_ui: Optional[int],
    user_comment_ui: Optional[str],
    action_type: str,  # "submit", "skip", "perfect"
) -> Generator[Tuple[Any, ...], None, None]:
    app_status["is_awaiting_feedback"] = False
    app_status["is_processing_something"] = True  # Now processing DSPy step
    log_msgs = [f"▶️ Resuming process after user interaction ('{action_type}')..."]
    yield _create_ui_update_tuple(
        app_status, new_log_messages=log_msgs
    )  # Hide feedback UI, show processing

    user_feedback_data: Optional[Dict[str, Any]] = None

    if action_type == "submit":
        if user_rating_ui is not None and user_comment_ui is not None:
            user_feedback_data = {
                "rating": int(user_rating_ui),
                "comment": user_comment_ui,
            }
            app_status["accumulated_log_entries"].append(
                f"🗣️ User Feedback Submitted: Rating [{user_feedback_data['rating']}/5], Comment: '{user_feedback_data['comment']}'"
            )
        else:
            app_status["accumulated_log_entries"].append(
                "⚠️ User submitted feedback, but rating or comment was empty. Proceeding without feedback for this step."
            )
    elif action_type == "skip":
        app_status["accumulated_log_entries"].append(
            "⏭️ User skipped providing feedback for this iteration."
        )
    elif action_type == "perfect":
        app_status["user_forced_stop"] = True
        app_status["accumulated_log_entries"].append(
            "✅ User marked current image as perfect! Stopping refinement."
        )
        app_status["is_processing_something"] = False  # No further processing
        yield _create_ui_update_tuple(app_status)
        return  # Stop the process here

    # Proceed to DSPy revision step
    yield from execute_dspy_revision_step(app_status, user_feedback_data)


def execute_dspy_revision_step(
    app_status: Dict[str, Any], user_feedback_data: Optional[Dict[str, Any]] = None
) -> Generator[Tuple[Any, ...], None, None]:
    app_status["is_processing_something"] = True  # Mark as busy for DSPy call
    log_msgs = [
        f"\n⚙️ Executing DSPy Revision for Iteration {app_status['current_iter_num']}..."
    ]
    yield _create_ui_update_tuple(app_status, new_log_messages=log_msgs)

    try:
        predictor = get_dspy_predictor(with_feedback=bool(user_feedback_data))

        # Construct arguments for the predictor
        dspy_args = {
            "desired_prompt": app_status["initial_prompt"],
            "current_image": app_status["current_dspy_image"],
            "current_prompt": app_status["current_prompt"],
        }
        if user_feedback_data:
            dspy_args["user_rating"] = user_feedback_data["rating"]
            dspy_args["user_comment"] = user_feedback_data["comment"]
            app_status["accumulated_log_entries"].append(
                f"🧠 Calling DSPy with user feedback: Rating {dspy_args['user_rating']}, Comment '{dspy_args['user_comment']}'"
            )
        else:
            app_status["accumulated_log_entries"].append(
                "🧠 Calling DSPy without explicit user feedback for this revision."
            )

        if app_status["current_dspy_image"] is None:
            app_status["accumulated_log_entries"].append(
                "❌ ERROR: Cannot run DSPy revision - `current_dspy_image` is missing in app state."
            )
            raise ValueError(
                "DSPy image missing in state, cannot proceed with revision."
            )

        result = predictor(**dspy_args)
        app_status["last_dspy_result"] = result

        if result.image_strictly_matches_desired_prompt:
            app_status["accumulated_log_entries"].append(
                "✅ DSPy: Image strictly matches desired prompt! Stopping refinement."
            )
            app_status["user_forced_stop"] = True
            app_status["current_prompt"] = (
                result.revised_prompt
            )  # Update to the prompt that matched
        else:
            app_status["current_prompt"] = result.revised_prompt
            app_status["accumulated_log_entries"].extend(
                [
                    f"🤖 DSPy AI Feedback: {result.feedback}",
                    f"↪️ DSPy Revised Prompt (for next iter): '{app_status['current_prompt']}'",
                ]
            )

    except Exception as e:
        error_msg = f"❌ Error during DSPy revision (Iter {app_status['current_iter_num']}): {str(e)}"
        app_status["user_forced_stop"] = True
        app_status["accumulated_log_entries"].append(error_msg)

    app_status["is_processing_something"] = False  # Done with DSPy call
    yield _create_ui_update_tuple(app_status)

    # Decide whether to continue to the next iteration or stop
    if (
        not app_status["user_forced_stop"]
        and app_status["current_iter_num"] < app_status["max_iter"]
    ):
        yield _create_ui_update_tuple(
            app_status,
            new_log_messages=[
                f"➡️ Proceeding to next iteration ({app_status['current_iter_num'] + 1})..."
            ],
        )
        yield from execute_iteration_step(
            app_status
        )  # This will increment current_iter_num
    else:
        app_status["is_processing_something"] = (
            False  # Ensure this is false on final stop
        )
        final_log_message = "🏁 Refinement process concluded."
        if (
            app_status["user_forced_stop"]
            and not result.image_strictly_matches_desired_prompt
        ):  # check result only if not None
            final_log_message = "🏁 Refinement process stopped by user or an error."
        elif app_status["current_iter_num"] >= app_status["max_iter"]:
            final_log_message = f"🏁 Max iterations ({app_status['max_iter']}) reached."

        yield _create_ui_update_tuple(app_status, new_log_messages=[final_log_message])


def execute_iteration_step(
    app_status: Dict[str, Any],
) -> Generator[Tuple[Any, ...], None, None]:
    if app_status["user_forced_stop"]:
        app_status["is_processing_something"] = False
        app_status["accumulated_log_entries"].append(
            "🛑 Process stopped by user or error."
        )
        yield _create_ui_update_tuple(app_status)
        return

    if app_status["current_iter_num"] >= app_status["max_iter"]:
        app_status["is_processing_something"] = False
        app_status["accumulated_log_entries"].append(
            f"🏁 Max iterations ({app_status['max_iter']}) reached."
        )
        yield _create_ui_update_tuple(app_status)
        return

    app_status["current_iter_num"] += 1
    app_status["is_processing_something"] = True

    log_msgs = [
        f"\n--- Iteration {app_status['current_iter_num']}/{app_status['max_iter']} ---",
        f"⏳ Generating image with prompt: '{app_status['current_prompt']}'",
    ]
    yield _create_ui_update_tuple(app_status, new_log_messages=log_msgs)

    try:
        dspy_image, pil_image = get_retro_diffusion_image(
            prompt=app_status["current_prompt"],
            input_palette_pil=app_status["input_palette_pil"],
            input_image_pil=app_status["input_image_pil"],
            strength=app_status["strength"],
            width=app_status["width"],
            height=app_status["height"],
            prompt_style=app_status["prompt_style"],
            seed=app_status["seed"],
        )
        app_status["current_dspy_image"] = dspy_image
        app_status["current_pil_image"] = pil_image

        gallery_caption = (
            f"Iter {app_status['current_iter_num']}: {app_status['current_prompt']}"
        )
        if len(gallery_caption) > 250:
            gallery_caption = gallery_caption[:247] + "..."
        app_status.get("accumulated_gallery_items", []).append(
            (pil_image, gallery_caption)
        )

        yield _create_ui_update_tuple(
            app_status, new_log_messages=["🖼️ Image generated successfully."]
        )

    except Exception as e:
        error_msg = f"❌ Error during image generation (Iter {app_status['current_iter_num']}): {str(e)}"
        app_status["user_forced_stop"] = True
        app_status["is_processing_something"] = False
        yield _create_ui_update_tuple(app_status, new_log_messages=[error_msg])
        return  # Stop the process here

    # Feedback decision point
    if app_status["feedback_enabled"]:
        app_status["is_awaiting_feedback"] = True
        app_status["is_processing_something"] = (
            False  # Paused, not actively processing API calls
        )
        yield _create_ui_update_tuple(
            app_status,
            new_log_messages=[
                "🔔 Paused for your feedback. Please use the feedback section below."
            ],
        )
        # The process now waits for feedback button clicks to call another function.
    else:
        # No feedback enabled, proceed directly to DSPy revision
        yield _create_ui_update_tuple(
            app_status,
            new_log_messages=["🤖 Auto-proceeding to AI revision (feedback disabled)."],
        )
        yield from execute_dspy_revision_step(app_status, user_feedback_data=None)


def start_main_refinement_process(
    initial_prompt_ui: str,
    max_iter_ui: int,
    palette_image_ui: Optional[Image.Image],
    input_image_ui: Optional[Image.Image],
    strength_ui: float,
    width_ui: int,
    height_ui: int,
    prompt_style_ui: str,
    seed_str_ui: str,
    feedback_enabled_ui: bool,
) -> Generator[Tuple[Any, ...], None, None]:
    app_status = _prepare_initial_state(
        initial_prompt_ui,
        max_iter_ui,
        palette_image_ui,
        input_image_ui,
        strength_ui,
        width_ui,
        height_ui,
        prompt_style_ui,
        seed_str_ui,
        feedback_enabled_ui,
    )
    app_status["is_processing_something"] = True  # Initial state is processing
    yield _create_ui_update_tuple(
        app_status, new_log_messages=["🚀 Process Starting..."]
    )

    if not app_status["initial_prompt"] or not app_status["initial_prompt"].strip():
        app_status["is_processing_something"] = False
        app_status["user_forced_stop"] = True
        yield _create_ui_update_tuple(
            app_status, new_log_messages=["❌ Error: Initial prompt cannot be empty."]
        )
        return

    if app_status["max_iter"] <= 0:
        app_status["is_processing_something"] = False
        app_status["user_forced_stop"] = True
        yield _create_ui_update_tuple(
            app_status,
            new_log_messages=["❌ Error: Max iterations must be a positive number."],
        )
        return

    log_msg_list = [
        f"📝 Initial Prompt: '{app_status['initial_prompt']}'",
        f"🔄 Max Iterations: {app_status['max_iter']}",
    ]
    if app_status["input_palette_pil"]:
        log_msg_list.append("🎨 Using provided palette image.")
    if app_status["seed"] is not None:
        log_msg_list.append(f"🌱 Using Seed: {app_status['seed']}")
    else:
        log_msg_list.append("🌱 Using Random Seed (or API default).")
    log_msg_list.append(
        f"💬 User Feedback Mode: {'Enabled' if app_status['feedback_enabled'] else 'Disabled'}"
    )
    yield _create_ui_update_tuple(app_status, new_log_messages=log_msg_list)

    # Start the first iteration
    yield from execute_iteration_step(app_status)


def save_llm_settings(api_key: str, model_dropdown: str, model_input: str, retro_token: str) -> tuple[str, str]:
    if not api_key:
        return gr.Warning("LLM API Key is required"), ""
    
    model_name = model_input if model_input.strip() else model_dropdown
    if not model_name:
        return gr.Warning("Please select a model or enter a custom model name"), ""
    
    try:
        configure_llm(api_key, model_name)
        if retro_token and retro_token.strip():
            configure_retro_diffusion(retro_token.strip())
        return gr.Info("Settings saved successfully!"), model_name
    except Exception as e:
        return gr.Error(f"Failed to configure settings: {str(e)}"), ""


# --- Gradio UI Definition and Event Wiring ---
if __name__ == "__main__":
    # Define available prompt styles
    PROMPT_STYLES = [
        "rd_fast__default",
        "rd_fast__retro",
        "rd_fast__simple",
        "rd_fast__detailed",
        "rd_fast__anime",
        "rd_fast__game_asset",
        "rd_fast__portrait",
        "rd_fast__texture",
        "rd_fast__ui",
        "rd_fast__item_sheet",
        "rd_fast__mc_texture",
        "rd_fast__mc_item",
        "rd_fast__character_turnaround",
        "rd_fast__1_bit",
        "rd_fast__no_style",
        "rd_plus__default",
        "rd_plus__retro",
        "rd_plus__watercolor",
        "rd_plus__textured",
        "rd_plus__cartoon",
        "rd_plus__ui_element",
        "rd_plus__item_sheet",
        "rd_plus__character_turnaround",
        "rd_plus__topdown_map",
        "rd_plus__topdown_asset",
        "rd_plus__isometric",
    ]

    # Define the initial structure of the app_status that gr.State will hold
    initial_app_status_dict_for_state = _prepare_initial_state(
        "", 3, None, None, 0.8, 150, 150, "rd_plus__retro", "", False
    )

    with gr.Blocks(
        theme=gr.themes.Glass(), css=".gradio-container {background-color: #f0f2f6;}"
    ) as demo:
        app_status_state_component = gr.State(
            value=initial_app_status_dict_for_state.copy()
        )

        gr.Markdown(
            """
            <div style=\"text-align: center;\">
                <h1>🎨 Advanced Prompt Refiner ✨</h1>
                <p>Craft the perfect prompt for your retro images. Iteratively refine your ideas with DSPy and see results instantly! Optionally provide feedback at each step.</p>
            </div>
            """
        )

        with gr.Accordion("🔑 API Configuration", open=False):
            with gr.Column():
                with gr.Group():
                    gr.Markdown("### 🤖 LLM Settings")
                    llm_api_key = gr.Textbox(
                        label="LLM API Key",
                        placeholder="Enter your API key here",
                        type="password",
                        elem_id="llm_api_key"
                    )
                    with gr.Row():
                        llm_model_dropdown = gr.Dropdown(
                            choices=[
                                "openai/gpt-4o-mini",
                                "anthropic/claude-3-opus-20240229",
                                "gemini/gemini-2.5-flash-preview-05-20"
                            ],
                            label="Suggested Models",
                            elem_id="llm_model_dropdown"
                        )
                        llm_model_input = gr.Textbox(
                            label="Custom Model Name",
                            placeholder="Or enter a custom model name",
                            elem_id="llm_model_input"
                        )
                    gr.Markdown(
                        "[Learn more about available models](https://dspy.ai/learn/programming/language_models)",
                        elem_id="llm_info_link"
                    )

                with gr.Group():
                    gr.Markdown("### 🎨 RetroDiffusion Settings")
                    retro_token = gr.Textbox(
                        label="RetroDiffusion API Token",
                        placeholder="Enter your RetroDiffusion token here (optional)",
                        type="password",
                        elem_id="retro_token"
                    )
                    gr.Markdown(
                        "Get your own token at [RetroDiffusion](https://retrodiffusion.ai)",
                        elem_id="retro_info_link"
                    )

                save_settings_button = gr.Button("💾 Save API Settings", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### ⚙️ Input Configuration")
                with gr.Group():
                    initial_prompt_input = gr.Textbox(
                        label="Your Creative Idea (Initial Prompt)",
                        placeholder="e.g., A pixel art cat wizard casting a powerful spell in a dark forest",
                        lines=4,
                        elem_id="initial_prompt",
                    )
                    max_iter_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Refinement Strength (Max Iterations)",
                        info="How many times should we try to improve the prompt?",
                    )
                    feedback_enabled_checkbox = gr.Checkbox(
                        label="Enable Iterative User Feedback",
                        value=False,
                        info="If checked, the process will pause after each image for your rating and comments.",
                    )

                with gr.Accordion("🎨 Advanced Options", open=False):
                    with gr.Row():
                        width_slider = gr.Slider(
                            minimum=64, maximum=512, value=150, step=8, label="Width"
                        )
                        height_slider = gr.Slider(
                            minimum=64, maximum=512, value=150, step=8, label="Height"
                        )

                    prompt_style_dropdown = gr.Dropdown(
                        choices=PROMPT_STYLES,
                        value="rd_plus__retro",
                        label="Prompt Style",
                        info="Select the style for image generation",
                    )

                    with gr.Row():
                        palette_image_input = gr.Image(
                            type="pil",
                            label="Color Palette (Optional)",
                            height=200,
                            width=200,
                            sources=["upload", "clipboard"],
                            elem_id="palette_image",
                        )
                        seed_input_text = gr.Textbox(
                            label="Seed (Optional)",
                            placeholder="e.g., 12345 or leave blank",
                            info="Use a specific seed for reproducible results.",
                            elem_id="seed_input",
                        )

                    with gr.Row():
                        input_image = gr.Image(
                            type="pil",
                            label="Input Image (Optional)",
                            height=200,
                            width=200,
                            sources=["upload", "clipboard"],
                            elem_id="input_image",
                        )
                        strength_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.8,
                            step=0.05,
                            label="Image Strength",
                            info="How much to preserve from the input image (if provided)",
                        )

                submit_button = gr.Button(
                    "🚀 Generate & Refine Image",
                    variant="primary",
                    elem_id="submit_button_custom",
                )

            with gr.Column(scale=3):
                gr.Markdown("### 🖼️ Generated Image & Final Prompt")
                with gr.Group():
                    output_image_display = gr.Image(
                        label="Your Retro Masterpiece! (Current/Final Image)",
                        height=380,
                        width=380,
                        interactive=False,
                        elem_id="output_image",
                    )
                    final_prompt_display = gr.Textbox(
                        label="✨ Final Polished Prompt",
                        interactive=False,
                        lines=3,
                        show_copy_button=True,
                        elem_id="final_prompt",
                    )

        with gr.Accordion(
            "✍️ Provide Your Feedback for Iteration X", visible=False
        ) as user_feedback_accordion:
            user_rating_slider = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=3,
                label="Your Rating (1=Poor, 5=Excellent)",
            )
            user_comment_input = gr.Textbox(
                label="Your Comments/Suggestions for This Image",
                lines=3,
                placeholder="e.g., 'More vibrant colors', 'Make the cat look angrier'",
            )
            with gr.Row():
                submit_user_feedback_button = gr.Button(
                    "Submit Feedback & Continue Revision"
                )
                skip_user_feedback_button = gr.Button(
                    "Skip Feedback & Continue Revision"
                )
                mark_as_perfect_button = gr.Button(
                    "✅ This is Perfect! Stop Here.", variant="stop"
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Iteration Gallery & Prompts")
                iteration_gallery_display = gr.Gallery(
                    label="Image Generation Journey",
                    show_label=True,
                    elem_id="iteration_gallery",
                    columns=[4],
                    object_fit="contain",
                    height="auto",
                    preview=True,
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📜 Detailed Prompt Refinement Log")
                history_log_display = gr.Textbox(
                    label="Step-by-step generation log:",
                    lines=12,
                    interactive=False,
                    show_copy_button=True,
                    elem_id="history_log",
                )

        gr.Markdown("---")

        ui_outputs_for_all_updates = [
            output_image_display,
            final_prompt_display,
            history_log_display,
            iteration_gallery_display,
            submit_button,
            initial_prompt_input,
            max_iter_slider,
            palette_image_input,
            seed_input_text,
            feedback_enabled_checkbox,
            user_feedback_accordion,
            user_rating_slider,
            user_comment_input,
            submit_user_feedback_button,
            skip_user_feedback_button,
            mark_as_perfect_button,
            app_status_state_component,
        ]

        submit_button.click(
            fn=start_main_refinement_process,
            inputs=[
                initial_prompt_input,
                max_iter_slider,
                palette_image_input,
                input_image,
                strength_slider,
                width_slider,
                height_slider,
                prompt_style_dropdown,
                seed_input_text,
                feedback_enabled_checkbox,
            ],
            outputs=ui_outputs_for_all_updates,
        )

        # Feedback button event handlers
        feedback_button_inputs = [
            app_status_state_component,
            user_rating_slider,
            user_comment_input,
        ]

        submit_user_feedback_button.click(
            fn=partial(handle_user_feedback_and_proceed, action_type="submit"),
            inputs=feedback_button_inputs,
            outputs=ui_outputs_for_all_updates,
        )
        skip_user_feedback_button.click(
            fn=partial(handle_user_feedback_and_proceed, action_type="skip"),
            inputs=feedback_button_inputs,  # Rating and comment are passed but ignored by logic for "skip"
            outputs=ui_outputs_for_all_updates,
        )
        mark_as_perfect_button.click(
            fn=partial(handle_user_feedback_and_proceed, action_type="perfect"),
            inputs=feedback_button_inputs,  # Rating and comment are passed but ignored by logic for "perfect"
            outputs=ui_outputs_for_all_updates,
        )

        

        # Wire up settings
        save_settings_button.click(
            fn=save_llm_settings,
            inputs=[llm_api_key, llm_model_dropdown, llm_model_input, retro_token],
            outputs=[gr.Text(visible=False), llm_model_input]
        )

        demo.css = """
        #initial_prompt textarea { font-size: 16px; }
        #submit_button_custom { font-size: 18px; padding: 10px 20px; }
        .gradio-container { font-family: 'Inter', sans-serif; }
        #history_log textarea { font-family: 'monospace'; font-size: 13px; }
        #final_prompt textarea { font-family: 'monospace'; font-size: 14px; }
        #palette_image { min-height: 210px; }
        #output_image { min-height: 390px; }
        #iteration_gallery .gallery-item { border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        #iteration_gallery .gallery-item img { border-radius: 6px 6px 0 0; }
        #iteration_gallery .gallery-item figcaption { background-color: rgba(0,0,0,0.7); color: white; padding: 8px; font-size: 12px; border-radius: 0 0 6px 6px; max-height: 100px; overflow-y: auto; }
        #llm_api_key, #retro_token { font-family: 'monospace'; }
        #llm_model_input { font-family: 'monospace'; }
        #llm_info_link, #retro_info_link { font-size: 12px; margin-top: -8px; color: #666; }
        .api-config-group { border: 1px solid #e0e0e0; padding: 15px; border-radius: 8px; margin-bottom: 15px; }
        .api-config-group .gr-button-primary { width: 100%; margin-top: 15px; }
        """
    demo.launch()
