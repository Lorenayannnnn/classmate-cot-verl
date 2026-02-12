

def parse_out_main_cot_output(main_pred):
    # Note this is only called when enable_thinking=True

    if "</think>" in main_pred:
        before_think, after_think = main_pred.split("</think>", 1)
        final_output = after_think
        end_with_think = True
    else:
        before_think = main_pred
        final_output = None
        end_with_think = False

    if "<think>" in before_think:
        main_cot = before_think.split("<think>", 1)[1]
        start_w_think = True
    else:
        main_cot = before_think
        start_w_think = False

    return main_cot, final_output, start_w_think, end_with_think