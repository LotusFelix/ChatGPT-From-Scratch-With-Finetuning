
"""
Data utility functions: dictionary creation and input formatting.
"""
import pandas as pd

def create_dictionary(df, max_examples=None):
    """
    Convert pandas DataFrame to a list of dicts with keys 'instruction','input','output'.
    Args:
        df (pd.DataFrame): DataFrame with columns 'instruction','input','output'.
        max_examples (int, optional): maximum number of examples to include.
    Returns:
        List[dict]: list of data entries.
    """
    data_list = []
    for idx, row in df.iterrows():
        if max_examples is not None and idx >= max_examples:
            break
        entry = {
            "instruction": row["instruction"],
            "input": row["input"],
            "output": row["output"],
        }
        data_list.append(entry)
    return data_list


def format_input(entry):
    """
    Format a single entry into an instruction-input prompt.
    """
    intro = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    instruction = f"

### Instruction:
{entry['instruction']}"
    input_text = entry.get("input", "")
    input_section = f"

### Input:
{input_text}" if input_text else ""
    return intro + instruction + input_section