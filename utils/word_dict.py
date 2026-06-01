SPECIALS = {
    "BOS": "<|startoftext|>",
    "EOS": "<|endoftext|>",
    "SCHEMA_B": "<SCHEMA>",
    "SCHEMA_E": "</SCHEMA>",
    "STATS_B": "<STATS>",
    "STATS_E": "</STATS>",
    "ROW_B": "<|reserved_token_4|>",
    "ROW_E": "<|reserved_token_5|>",
    "COL_SEP": "|",
    "ASSN": "=",
    "EOT": "<eot_id>",
    "U_ST": "<|start_header_id|>user<|end_header_id|>",
    "A_ST": "<|start_header_id|>assistant<|end_header_id|>",
    "MASK": "<|mdm_mask|>",
    "NUMBER": "<|reserved_token_6|>"
}

NEW_SPECIALS = {
    "SCHEMA_B": "<SCHEMA>",
    "SCHEMA_E": "</SCHEMA>",
    "STATS_B": "<STATS>",
    "STATS_E": "</STATS>",
    "ROW_B": "<ROW>",
    "ROW_E": "</ROW>",
}

TOKENS_CONVERT = {
    "<|reserved_token_0|>": "<SCHEMA>",
    "<|reserved_token_1|>": "</SCHEMA>",
    "<|reserved_token_2|>": "<STATS>",
    "<|reserved_token_3|>": "</STATS>",
    "<|reserved_token_4|>": "<ROW>",
    "<|reserved_token_5|>": "</ROW>",
}