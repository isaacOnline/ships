from typing import NamedTuple, Any, Tuple

import tensorflow as tf


class TrainTranslator(tf.keras.Model):
    """
    Class implementing a single-layer stacked GRU model. Adapted from the below tutorial:
    https://www.tensorflow.org/text/tutorials/nmt_with_attention

    """
    def __init__(self, units, input_series_length, num_input_variables, output_series_length, num_output_variables):
        super().__init__()
        # Build the encoder and decoder
        self.input_series_length = input_series_length
        self.num_input_variables = num_input_variables
        self.output_series_length = output_series_length
        self.num_output_variables = num_output_variables
        self.encoder = Encoder(num_input_variables,
                               units)
        self.decoder = Decoder(num_output_variables,
                               units)

        self.shape_checker = ShapeChecker()


    def _loop_step(self, input_token, target_token, enc_output, dec_state):
        """
        Make one timestep worth of predictions using the decoder and calculate loss for the timestep

        :param input_token: Input data
        :param target_token: Ground truth data
        :param enc_output: Output from encoder
        :param dec_state: Previous state for decoder
        :return:
        """
        # Run the decoder one step.
        decoder_input = DecoderInput(new_tokens=input_token,
                                     enc_output=enc_output)

        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)
        self.shape_checker(dec_result.predictions, ('batch', 't1', 'predictions'))
        self.shape_checker(dec_result.attention_weights, ('batch', 't1', 's'))
        self.shape_checker(dec_state, ('batch', 'dec_units'))

        # `self.loss` returns the total for non-padded tokens
        y = target_token
        y_pred = dec_result.predictions
        step_loss = self.loss(y, y_pred)

        return y_pred, step_loss, dec_state

    @tf.function
    def train_step(self, inputs):
        """
        Make predictions using the model, calculate loss, and perform the optimization step

        :param inputs:
        :return:
        """
        input_text, target_text = inputs

        max_target_length = tf.shape(target_text)[1]

        with tf.GradientTape() as tape:
            # Encode the input
            enc_output, enc_state = self.encoder(input_text)
            self.shape_checker(enc_output, ('batch', 's', 'enc_units'))
            self.shape_checker(enc_state, ('batch', 'enc_units'))

            # Initialize the decoder's state to the encoder's final state.
            # This only works if the encoder and decoder have the same number of
            # units.
            dec_state = enc_state
            loss = tf.constant(0.0)

            for t in tf.range(max_target_length):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder.
                # 2. The target for the decoder's next prediction.

                if t == 0:
                    decoder_input = input_text[:, -1:, :self.num_output_variables]
                else:
                    decoder_input = target_text[:, t-1:t, :]

                decoder_output = target_text[:, t:t + 1, :]

                y_pred, step_loss, dec_state = self._loop_step(decoder_input, decoder_output, enc_output, dec_state)
                loss = loss + step_loss

            # Average the loss over all non padding tokens.
            average_loss = loss / tf.cast(max_target_length, tf.float32)

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        # Return a dict mapping metric names to current value
        return {'loss': average_loss}

    @tf.function
    def test_step(self, data):
        """
        Wrapper for prediction, which returns the loss instead of predictions

        :param data: Dataset to predict
        :return:
        """
        x, y = data
        y_hat = self.predict(x)
        average_loss = self.loss(y, y_hat)

        return {'loss': average_loss}

    @tf.function
    def predict(self, input_text):
        """
        Make a set of predictions for the input text

        :param input_text: Text to make predictions for
        :return:
        """
        enc_output, enc_state = self.encoder(input_text)

        dec_state = enc_state
        new_tokens = input_text[:,-1:,:self.num_output_variables]
        result_tokens = tf.TensorArray(tf.float64, size=1, dynamic_size=True)

        for t in tf.range(self.output_series_length):
            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output)

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            new_tokens = dec_result.predictions
            # Collect the generated tokens
            result_tokens = result_tokens.write(t, tf.cast(tf.squeeze(new_tokens),tf.float64))

        result_tokens = result_tokens.stack()
        result_tokens = tf.transpose(result_tokens, [1, 0, 2])

        return result_tokens


class Encoder(tf.keras.layers.Layer):
    """
    Class for the encoder portion of the model
    """
    def __init__(self, num_input_variables, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.num_input_variables = num_input_variables

        # The GRU RNN layer processes those vectors sequentially.
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       # Return the sequence and state
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        """
        Process the input tokens, returning the encoder's predictions and state

        :param tokens:
        :param state: Any
        :return:
        """
        shape_checker = ShapeChecker()
        shape_checker(tokens, ('batch', 's', 'num_input_variables'))


        # 3. The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        #    state shape: (batch, enc_units)
        output, state = self.gru(tokens, initial_state=state)
        shape_checker(output, ('batch', 's', 'enc_units'))
        shape_checker(state, ('batch', 'enc_units'))

        # 4. Returns the new sequence and its state.
        return output, state


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Class implementing attention mechanism, based on the below paper:
    https://arxiv.org/pdf/1409.0473.pdf
    """
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value):
        """
        Call the attention mechanism

        :param query: RNN output
        :param value: Encoder output
        :return:
        """
        shape_checker = ShapeChecker()
        shape_checker(query, ('batch', 't', 'query_units'))
        shape_checker(value, ('batch', 's', 'value_units'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        shape_checker(w2_key, ('batch', 's', 'attn_units'))

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            return_attention_scores=True,
        )
        shape_checker(context_vector, ('batch', 't', 'value_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights


class DecoderInput(NamedTuple):
    new_tokens: Any
    enc_output: Any


class DecoderOutput(NamedTuple):
    predictions: Any
    attention_weights: Any


class Decoder(tf.keras.layers.Layer):
    """
    Decoder portion of model
    """
    def __init__(self, num_output_variables, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.num_output_variables = num_output_variables

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the predictions for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.num_output_variables)

    def call(self,
             inputs: DecoderInput,
             state=None) -> Tuple[DecoderOutput, tf.Tensor]:
        """
        Call the decoder object

        :param inputs: Output sequence
        :param state: Previous state for decoder
        :return:
        """
        shape_checker = ShapeChecker()
        shape_checker(inputs.new_tokens, ('batch', 't', 'num_output_variables'))
        shape_checker(inputs.enc_output, ('batch', 's', 'enc_units'))
        if state is not None:
            shape_checker(state, ('batch', 'dec_units'))

        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(inputs.new_tokens, initial_state=state)

        shape_checker(rnn_output, ('batch', 't', 'dec_units'))
        shape_checker(state, ('batch', 'dec_units'))

        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attention_weights = self.attention(query=rnn_output, value=inputs.enc_output)
        shape_checker(context_vector, ('batch', 't', 'dec_units'))
        shape_checker(attention_weights, ('batch', 't', 's'))

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Dense layer
        attention_vector = self.Wc(context_and_rnn_output)
        shape_checker(attention_vector, ('batch', 't', 'dec_units'))

        # Step 5. Generate predictions:
        predictions = self.fc(attention_vector)
        shape_checker(predictions, ('batch', 't', 'num_output_variables'))

        return DecoderOutput(predictions, attention_weights), state


class ShapeChecker():
    """
    Class for validating that the shapes of tensors used by a model do not change
    """
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        if isinstance(names, str):
            names = (names,)

        shape = tf.shape(tensor)
        rank = tf.rank(tensor)

        if rank != len(names):
            raise ValueError(f'Rank mismatch:\n'
                             f'    found {rank}: {shape.numpy()}\n'
                             f'    expected {len(names)}: {names}\n')

        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]

            if (broadcast and new_dim == 1):
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"    found: {new_dim}\n"
                                 f"    expected: {old_dim}\n")