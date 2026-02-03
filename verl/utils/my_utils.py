

def parse_out_main_cot_output(main_pred):
    main_cot, final_output = None, None

    # take the string between <think> and </think> in main_pred
    # think_pattern = r"<think>(.*?)</think>"
    # think_match = re.search(think_pattern, main_pred, re.DOTALL)
    # if think_match:
    #     main_cot = think_match.group(1).strip()
    #     # take things after </think>
    #     after_think_pattern = r"</think>(.*)"
    #     after_think_match = re.search(after_think_pattern, main_pred, re.DOTALL)
    #     if after_think_match:
    #         final_output = after_think_match.group(1).strip()
    #     else:
    #         final_output = None
    # else:
    #     # Fall back to take things after <think>
    #     after_think_pattern = r"<think>(.*)"
    #     after_think_match = re.search(after_think_pattern, main_pred, re.DOTALL)
    #     if after_think_match:
    #         main_cot = after_think_match.group(1).strip()

    if "</think>" in main_pred:
        before_think, after_think = main_pred.split("</think>", 1)
        final_output = after_think or None
    else:
        before_think = main_pred

    if "<think>" in before_think:
        start_w_think = True
        main_cot = before_think.split("<think>", 1)[1]
    else:
        start_w_think = False
        main_cot = before_think

    assert main_cot is not None, f"Got main_cot is None when parsing main_pred: {main_pred}"
    return main_cot, final_output, start_w_think