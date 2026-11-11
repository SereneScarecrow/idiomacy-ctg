"""
Prompts are text files (prompts/<name>.txt) with {{variable}} placeholders, the same
placeholder syntax as in Langfuse. The file name is the prompt name and carries the version
(idiomatic_low_v1); a file is never edited after it has been used, a change is a new version.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str                   # file name without extension
    template: str               # text with {{placeholders}}
    variables: tuple[str, ...]  # names of the placeholders, in order of appearance

    def render(self, **values) -> str:
        """
        Fills the placeholders. A variable without a value is an error; values that the
        template does not use are ignored, so several prompts can share one set of values.
        """
        missing = [name for name in self.variables if name not in values]
        if missing:
            raise ValueError(f"Prompt {self.name!r}: no value for {missing}")
        return _PLACEHOLDER.sub(lambda match: str(values[match.group(1)]), self.template)


def load_prompt(path) -> Prompt:
    """
    Reads a prompt file (UTF-8, with or without BOM). Line endings are unified and blank
    lines around the text are removed, so the template does not depend on the operating
    system or the editor.
    """
    path = Path(path)
    template = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip("\n")
    return Prompt(
        name=path.stem,
        template=template,
        variables=tuple(dict.fromkeys(_PLACEHOLDER.findall(template))),
    )