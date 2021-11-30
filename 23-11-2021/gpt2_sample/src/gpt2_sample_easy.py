import gradio as gr
from transformers import pipeline, set_seed

MAX_LEN = 50
NUM_RETURN_SEQ = 5
SEED = 22333
TEXT_GEN = None
DO_SAMPLE = True


def get_text_generator(task="text-generation", model="gpt2"):
    return pipeline(task, model=model)


def get_gradio_fn(sentence, max_len, do_sample, num_return_sequences):
    num_sequences = int(num_return_sequences)
    if TEXT_GEN is not None:
        set_seed(SEED)
        generated_res = TEXT_GEN(sentence,
                                 max_length=max_len,
                                 do_sample=do_sample,
                                 num_return_sequences=num_sequences)

        if len(generated_res) <= num_sequences:
            num_sequences = len(generated_res)

        generated_sens = ""
        for i in range(num_sequences):
            generated_sens += 'Generated Sentences ' + str(i + 1) + ': ' + generated_res[i].get(
                'generated_text') + '\r\n'
        print("Initial Sentence: %s\r\n%s\r\n\r\n" %
              (sentence, generated_sens))
        return generated_sens
    else:
        return "Error occur when initializing the generator !"


if __name__ == "__main__":
    # init model
    TEXT_GEN = get_text_generator()

    # gradio
    gradio_model = gr.Interface(
        fn=get_gradio_fn,
        inputs=[
            gr.inputs.Textbox(lines=2, default="", label="Sentence"),
            gr.inputs.Number(default=MAX_LEN, label="MaxLen"),
            gr.inputs.Checkbox(default=DO_SAMPLE, label="DoSample"),
            gr.inputs.Number(default=NUM_RETURN_SEQ, label="Num Of Returned Sequences"),
        ],
        outputs="text"
    )
    gradio_model.launch()

# sample
# These paper lanterns are adorable! The colors are

# Do not buy these!
