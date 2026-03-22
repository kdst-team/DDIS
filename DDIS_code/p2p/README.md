This directory was previously committed as a Git submodule entry without a
matching `.gitmodules` definition, which breaks fresh checkouts in GitHub
Actions.

It is now kept as a normal directory so the repository can be cloned and
checked out reliably.

If you want to run `DDIS_code/main_ddis.py`, add the expected Prompt-to-Prompt
source files here, including:

- `ptp_utils_distG.py`
- `prompt_to_prompt.py`
