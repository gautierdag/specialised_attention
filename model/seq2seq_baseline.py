from machine.models import EncoderRNN, DecoderRNN, Seq2seq


def get_baseline_model(src, tgt, max_len=50, hidden_size=50, embedding_size=100):
    # Initialize model
    encoder = EncoderRNN(len(src.vocab),
                         max_len,
                         hidden_size,
                         embedding_size,
                         rnn_cell='gru')
    decoder = DecoderRNN(len(tgt.vocab),
                         max_len,
                         hidden_size,
                         rnn_cell='gru',
                         eos_id=tgt.eos_id,
                         sos_id=tgt.sos_id)
    seq2seq = Seq2seq(encoder, decoder)

    # # initialize weights
    # for param in seq2seq.parameters():
    #     param.data.uniform_(-0.08, 0.08)

    return seq2seq
