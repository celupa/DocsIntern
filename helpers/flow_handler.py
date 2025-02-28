import re
from typing import Tuple
from helpers.model_selector import ModelSelector
from helpers.rag_handler import RAGHandler
from helpers import cfg as c
from helpers import custom_functions as cfoos


class FlowHandler:
    """Coordinates messages to LLM and DB."""

    def __init__(
            self,
            model: ModelSelector,
            rag_handler: RAGHandler,
    ) -> None:
        try:
            self.model = model
            self.rag_handler = rag_handler
            # print(f"---{datetime.now()} | FlowHandler ready")
            cfoos.LOG.info("FlowHandler ready")
        except:
            # print(f"---{datetime.now()} | Failed to initialize FlowHandler. Contact dev.")
            cfoos.LOG.info("Failed to initialize FlowHandler. Contact dev.")

    def route_prompt(
            self,
            prompt: str,
            singleton: bool=False,
            sysprompt: str=None,
            extend_messages: bool=True
            ) -> Tuple[str, str] | Tuple[str, None]:
        """If the prompt contains 'db: ', search database, else prompt llm directly."""
        data_info = None
        # query database if flag in prompt
        if prompt.strip().startswith("db:"):
            db = c.DB_PATH / "chroma.sqlite3"
            if not db.exists():
                response = "Parse your Data Path before querrying the database."
                return response, data_info
            form_prompt = prompt.strip().split("db:")[1].strip()
            # print(f"---{datetime.now()} | Searching database...")
            cfoos.LOG.info("Searching database...")
            rag_retrievals = self.rag_handler.retrieve_vdata(form_prompt)
            valid_retrievals = self.assess_retrieval(form_prompt, rag_retrievals)
            response, data_info = self.summarize_retrievals(prompt, valid_retrievals)
            if response:
                return response, data_info
            # if there were no valid retrievals prompt llm to inform retreival failure
            sysp = f"""The database has returned no response for the following user prompt. 
            If you can't answer the user prompt yourself, briefly inform the user that you
            couldn't find the requested information in the database"""
            response = self.model.prompt_llm(
                prompt,
                singleton=True,
                sysprompt=sysp,
                extend_messages=True
            )
            return response, data_info
        # allow user free interaction with the llm
        response = self.model.prompt_llm(prompt, singleton, sysprompt, extend_messages)
        return response, data_info

    def assess_retrieval(
            self,
            prompt: str,
            rag_retrievals: dict
            ) -> dict:
        """Go over db retrievals and flag which one responds to user query."""
        # added the  for mardown formatting in gradio
        valid_retrievals = {
            "folders": [],
            "files": [],
            "quotes": []
        }
        # go over retrieved content
        for metadata in rag_retrievals["metadatas"][0]:
            snippet = metadata["content"]
            folder = re.search(r"folder:\s*[^,]+", snippet).group(0)
            file = re.search(r"file:\s*[^,]+", snippet).group(0)
            quote = re.search(r"text:\s*.*", snippet, re.DOTALL).group(0).split("text: ")[1]
            sysp = """Respond ONLY with 'True' if the provided snippet contains details
            that might help in answering the user's prompt, otherwise respond with 'False'.
            """
            comparison_prompt = f"""user prompt: {prompt}
            snippet: {snippet}
            """
            response = self.model.prompt_llm(
                prompt=comparison_prompt,
                sysprompt=sysp,
                singleton=True,
                extend_messages=False)
            # print(f"{prompt} ||| {response} ||| {folder} ||| {file} ||| {snippet}")
            # store quotes and sources if any retrievals match user prompt
            if "True" in response:
                quote = "quote: '..." + quote + "...'"
                valid_retrievals["folders"].append(folder)
                valid_retrievals["files"].append(file)
                valid_retrievals["quotes"].append(quote)
                # cpu llm doesn't have enough bandwith for full blown summarization
                if self.model.device == "cpu":
                    return valid_retrievals
        return valid_retrievals

    def summarize_retrievals(self, prompt: str, valid_retrievals: dict) -> Tuple[str, str] | Tuple[None, None]:
        """If any, summarize DB retreivals."""
        if len(valid_retrievals["quotes"]) > 0:
            # cpu won't handle much context, we limit material provided for summarization
            summary_content = ", ".join(valid_retrievals["quotes"]),
            sysp = f"""{self.model.system_prompt}. Additonally,
            Summarize the snippets provided within 400 characters, 
            in the language provided by the user prompt.
            Provide only the summary with no additonal commentary."""
            summarize_prompt = f"""user prompt: {prompt}
            snippets: {summary_content}"""
            response = self.model.prompt_llm(
                summarize_prompt,
                singleton=True,
                sysprompt=sysp,
                extend_messages=False)
            self.model.extend_conversation(prompt, response)
            # format sources nicely
            data_info = self.format_sources(valid_retrievals)
            return response, data_info
        return None, None

    def format_sources(self, valid_retrievals: dict) -> str:
        """Format valid responses provided by assess_retrieval."""
        formatted_sources = "\nSources:"
        sections = list(valid_retrievals.keys())
        nvals = len(valid_retrievals[sections[0]])
        for i in range(nvals):
            formatted_sources += "\n---\n"
            for section in sections:
                formatted_sources += f""">{valid_retrievals[section][i]}\n"""
        return formatted_sources

    def prompt_llm(
            self,
            prompt: str,
            singleton: bool=False,
            sysprompt: str=None,
            extend_messages: bool=True
            ) -> Tuple[str, str] | Tuple[str, None]:
        """Prompt the correct model based on hardware availability."""
        response, data_info = self.route_prompt(prompt, singleton, sysprompt, extend_messages)
        return response, data_info
