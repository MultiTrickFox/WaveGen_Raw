def main():

    import config

    from model import load_model
    model = load_model()
    while not model:
        config.model_path = input('valid model: ')
        model = load_model()

    from data import load_data, split_data, file_output
    d = load_data(frames=not config.attention_only)
    # d, _ = split_data(d)
    # from random import shuffle
    # shuffle(d)

    for i,seq in enumerate(d[:config.hm_wav_gen]):

        from model import respond_to
        _, seq = respond_to(model, [seq], training_run=False, extra_steps=config.hm_extra_steps)
        seq = seq.detach()
        if config.use_gpu:
            seq = seq.cpu()
        seq = seq.numpy()

        file_output(f'{config.output_file}{i}', seq)

if __name__ == '__main__':
    main()