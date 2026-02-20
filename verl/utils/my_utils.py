
import torch


def parse_out_main_cot_output(
    main_pred,
    main_pred_token_ids,
    tokenizer,
    think_open_id,
    think_close_id,
):
    """Parse CoT and final output from a model response using token-level boundaries.

    Locates <think> and </think> directly in ``main_pred_token_ids`` so that
    the returned ``cot_start`` / ``cot_end`` indices are correctly aligned with
    ``data.batch["responses"]``.  ``main_cot`` is decoded from those exact token
    positions (no re-encoding round-trip).

    Note: only called when enable_thinking=True.

    Parameters
    ----------
    main_pred            : str   – decoded response string (used as fallback text
                                   for main_cot when </think> is absent)
    main_pred_token_ids  : Tensor or list – full response token IDs
    tokenizer            : tokenizer instance
    think_open_id        : int – token ID of <think>
    think_close_id       : int – token ID of </think>

    Returns
    -------
    main_cot         : str   – CoT text
    final_output     : str | None – text after </think>, or None if not found
    start_with_think : bool
    end_with_think   : bool
    cot_start        : int   – index of first CoT token in the response tensor
    cot_end          : int | None – exclusive end index (== index of </think>),
                                    or None when </think> was not found;
                                    caller should clamp with the valid response length
    """
    if not isinstance(main_pred_token_ids, torch.Tensor):
        main_pred_token_ids = torch.tensor(main_pred_token_ids)

    open_positions  = (main_pred_token_ids == think_open_id).nonzero(as_tuple=True)[0]
    close_positions = (main_pred_token_ids == think_close_id).nonzero(as_tuple=True)[0]

    if len(open_positions) > 0:
        first_open = open_positions[0].item()
        # TODO haha include <think> in CoT?
        cot_start = first_open + 1  # CoT begins right after <think>
        # cot_start = first_open  # CoT begins AT <think>
        start_with_think = True

        # First </think> that appears after the opening tag
        close_after_open = close_positions[close_positions > first_open]
        if len(close_after_open) > 0:
            # TODO haha include <think> in CoT?
            cot_end = close_after_open[0].item()  # CoT ends right before </think>
            # cot_end = close_after_open[0].item() + 1    # CoT ends right AT </think>
            end_with_think = True
            final_output = tokenizer.decode(
                main_pred_token_ids[cot_end + 1:].tolist(), skip_special_tokens=True
            )
            main_cot = tokenizer.decode(
                main_pred_token_ids[cot_start:cot_end].tolist(), skip_special_tokens=True
            )
        else:
            # No </think> found after <think>.  cot_end=None signals to the caller
            # to clamp using the valid response length from response_mask.
            # main_cot falls back to text splitting because we cannot determine the
            # valid end boundary (vs. padding) from token IDs alone here.
            cot_end = None
            end_with_think = False
            final_output = None
            main_cot = main_pred.split("<think>", 1)[1] if "<think>" in main_pred else main_pred
    else:
        cot_start = 0
        start_with_think = False
        cot_end = None
        end_with_think = False
        final_output = None
        main_cot = main_pred

    return main_cot, final_output, start_with_think, end_with_think, cot_start, cot_end
