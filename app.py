from typing import Tuple
from pathlib import Path
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
import gradio as gr
from helpers import cfg as c
from helpers.model_selector import ModelSelector
from helpers.rag_handler import RAGHandler
from helpers.evaluator import Evaluator
from helpers.tuner import Tuner
from helpers.flow_handler import FlowHandler
from helpers import custom_functions as cfoos


# redirect logging to log buffer
LOG_BUFFER = io.StringIO()
stream_handler = logging.StreamHandler(LOG_BUFFER)
cfoos._set_log_stream(cfoos.LOG, stream_handler)
# prepare a separate thread to timeout error coming from gradio
EXECUTOR = ThreadPoolExecutor()
# preconfigure app
MODEL_SELECTOR = ModelSelector(c.MODEL_CONFIG)
RAG_HANDLER = None
FLOW_HANDLER = None


def update_console():
    """Output "logs" to console, but truncate if too many lines."""
    lines = LOG_BUFFER.getvalue().split("\n")
    nlines = len(LOG_BUFFER.getvalue())
    # limit output lines
    if nlines >= 46:
        console_lines = lines[-46:]
        console = "\n".join(console_lines) 
        LOG_BUFFER.truncate(0)
        LOG_BUFFER.seek(0)
        LOG_BUFFER.write(console)
        return LOG_BUFFER.getvalue()
    return LOG_BUFFER.getvalue()

def prime_flow(data_path: str=None, tune: str=None) -> None:
    """
    Initialize application depending on state.
    params:
        data_path: path to data
        tune: if true, app will find most optimal params (rag retrieval, chunks...)
        """
    global RAG_HANDLER
    global FLOW_HANDLER
    # if data has been parsed before, init_config.txt will contain previous path
    # this is a precursor to future updates
    init_config_path = Path(c.APP_DIR / "init_config.txt")
    # if data_path and tuning has been provided by the user, then begin parsing flow
    if data_path:
        RAG_HANDLER = RAGHandler(
            db_path=c.DB_PATH,
            data_path=data_path,
            supported_extensions=c.SUPPORTED_EXTENSIONS,
            collection_config=c.COLLECTION_CONFIG,
            collection_name=c.DB_COLLECTION,
            model=MODEL_SELECTOR
            )
        data = RAG_HANDLER.extract_data()
        chunks = RAG_HANDLER.get_chunks(
            extracted_data=data, 
            chunk_size_factor=1.0,
            chunk_overlap_factor=0.2)
        RAG_HANDLER.populate_vectdb(chunks)
        if tune == "Yes":
            # get a baseline evaluation
            evaluator = Evaluator(MODEL_SELECTOR, RAG_HANDLER)
            df = evaluator.get_eval_df()
            eval_df = evaluator.generate_eval_prompts(
                df=df,
                read_local_df=False,
                save_df=False
                )
            anal_df = evaluator.prompt_db(eval_df)
            evaluator.analyze_retrievals(anal_df)
            tuner = Tuner(
                model=MODEL_SELECTOR,
                rag_handler=RAG_HANDLER,
                evaluator=evaluator,
                tuning_config=c.TUNING_MODIFIERS,
                )
            # tune baseline evaluation
            tuning_df = tuner.tune_retrieval(save_df=False)
            # ftm, we default to speed 
            tuner.select_user_config(tuning_df, manual_opt="speed")
    # if user didn't opt for parsing, fetch existing config file and get RAGHandler
    if init_config_path.exists():
        with open(init_config_path, "r") as f:
            stored_data_path = f.read()
        RAG_HANDLER = RAGHandler(
            db_path=c.DB_PATH,
            data_path=stored_data_path,
            supported_extensions=c.SUPPORTED_EXTENSIONS,
            collection_config=c.COLLECTION_CONFIG,
            collection_name=c.DB_COLLECTION,
            model=MODEL_SELECTOR
            )
    else:
        RAG_HANDLER = RAGHandler(
            db_path=c.DB_PATH,
            data_path="None",
            supported_extensions=c.SUPPORTED_EXTENSIONS,
            collection_config=c.COLLECTION_CONFIG,
            collection_name=c.DB_COLLECTION,
            model=MODEL_SELECTOR
            )
    FLOW_HANDLER = FlowHandler(
        model=MODEL_SELECTOR,
        rag_handler=RAG_HANDLER
        )
        
def interaction_on(*args: str) -> None:
    """Turn widget interaction on."""
    return [gr.update(interactive=True) for _ in range(len(args))]

def interaction_off(*args: str) -> None:
    """Turn widget interaction off."""
    return [gr.update(interactive=False) for _ in range(len(args))]

def format_path(text: str) -> str | None:
    # format linux/mac path
    linux_mac_regex = r'/[^\s:*?"<>|]+'  
    # format windows paths
    windows_regex = r'[a-zA-Z]:\\[^\s:*?"<>|]+' 
    # format unc network path
    unc_regex = r'\\\\[^\s:*?"<>|]+'  
    pattern = f"({linux_mac_regex})|({windows_regex})|({unc_regex})"
    match = re.search(pattern, text)
    if match:
        return next(filter(None, match.groups())) 
    return "None"

def validate_path(data_path=None, tune: str=None) -> Tuple[dict, dict]:
    """Validate data path and control app flow."""
    global PARSING_STATUS
    path_warning = "Hey, you! Please check the provided path exists, contains valid files and is formatted correctly..."
    try:
        data_path = format_path(data_path)
        if not Path(data_path).exists():
            data_path = gr.update(label=path_warning, value="")
            return data_path
        PARSING_STATUS = "working"
        future = EXECUTOR.submit(prime_flow, data_path, tune)
        future.result()
        data_path = gr.update(label="Data Path", value="")
        PARSING_STATUS = "done"
        return data_path
    except ValueError:
        PARSING_STATUS = "done"
        data_path = gr.update(label=path_warning, value="")
        return data_path

def chat(prompt, history):
    """Support gradio with history building."""
    response, data_info = FLOW_HANDLER.prompt_llm(prompt)
    if history is None:
        history = []
    if data_info:
        response = f"{response}\n{data_info}"
    history.append(({"role": "user", "content": prompt}))
    history.append(({"role": "assistant", "content": response}))
    # empty chat input text
    chat_input = gr.Textbox(value="")
    # one history for the chatbot and one for the state
    return history, history, chat_input

# init default logic
prime_flow()


# disable tuning for CPUs
tune_state = False if MODEL_SELECTOR.device == "cpu" else True
tune_text = "Tune (CUDA required)" if MODEL_SELECTOR.device == "cpu" else "Tune"
# setup gradio
with gr.Blocks() as app:
    # data parse logic and chatbot
    with gr.Row():
        with gr.Column(scale=1):
            data_path = gr.Textbox(label="Data Path")
            tune = gr.Dropdown(
                ["No", "Yes"], 
                label=tune_text, 
                value="No",
                interactive=tune_state
                )
            parse_button = gr.Button("Parse")
            chatbot = gr.Chatbot(label="Chatbot", type="messages")
            chat_input = gr.Textbox(show_label=False, placeholder="Ask me something")
            chat_button = gr.Button("Chat")
            parse_button.click(interaction_off, 
                               inputs=[parse_button, chat_input, chat_button], 
                               outputs=[parse_button, chat_input, chat_button]
                               ).then(validate_path, 
                                      inputs=[data_path, tune], 
                                      outputs=[data_path]
                                      ).then(interaction_on, 
                                             inputs=[parse_button, chat_input, chat_button], 
                                             outputs=[parse_button, chat_input, chat_button]
                                             )
            state = gr.State([])
            chat_button.click(chat, inputs=[chat_input, state], outputs=[chatbot, state, chat_input])
        with gr.Column(scale=1):
            console = gr.Code(
            interactive=False,
            lines=46,
            max_lines=46,
            label="Console",
            value=""
        )
    timer = gr.Timer(0.1, active=True)
    timer.tick(update_console, outputs=console)

app.launch(inbrowser=True)
