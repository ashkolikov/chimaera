import tensorflow as tf
import tensorflow.keras.backend as K
#import keras_nlp

class MainModel():
    def __init__(self,
                 data,
                 decoder,
                 latent_dim = 64,
                 filters = [64, 128, 256, 512, 64],
                 kernel_sizes = [19, 9, 5, 3, 3],
                 dilation_rates = [1, 2, 16, 1, 1],
                 n_residual_blocks = 0,
                 n_attention_blocks = 0,
                 residual_block_filters = [256, 128],
                 residual_block_kernel_sizes = [5, 5],
                 residual_block_dilation_rate_factors = [4, 4],
                 querry_dim = 32,
                 n_heads = 8,
                 dropout_rates = [0, 0.1, 0.15, 0.15, 0.15],
                 residual_block_dropout_rates = [0, 0],
                 attention_block_dropout_rate = 0.1):


        inp = tf.keras.Input((data.dna_len, 4))

        x = tf.keras.layers.Conv1D(filters[0], kernel_sizes[0],  dilation_rate=dilation_rates[0],  padding='same', kernel_initializer='he_normal')(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(dropout_rates[0])(x)

        pooling_range = int(np.log2(data.dna_len)) - 8
        for block in range(pooling_range-2):
            x = tf.keras.layers.Conv1D(filters[1], kernel_sizes[1], dilation_rate=dilation_rates[1], padding='same', kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Dropout(dropout_rates[1])(x)

        x = tf.keras.layers.Conv1D(filters[1], kernel_sizes[1], dilation_rate=dilation_rates[1], padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x_ = x
        for block in range(n_residual_blocks):
            x = tf.keras.layers.Conv1D(residual_block_filters[0], residual_block_kernel_sizes[0], padding='same', dilation_rate=residual_block_dilation_rate_factors[0]*(block+1), kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Dropout(residual_block_dropout_rates[0])(x)

            x = tf.keras.layers.Conv1D(residual_block_filters[1], residual_block_kernel_sizes[1], padding='same', dilation_rate=residual_block_dilation_rate_factors[1]*(block+1), kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(residual_block_dropout_rates[1])(x)
            x = tf.keras.layers.Add()([x_, x])
            x = tf.keras.layers.ReLU()(x)
            x_ = x
        
        x_ = x
        atts = []
        for i in range(n_attention_blocks):
            x = x + keras_nlp.layers.SinePositionEncoding()(x)
            x, att = tf.keras.layers.MultiHeadAttention(n_heads, querry_dim)(x, x, return_attention_scores=True)
            atts.append(att)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(attention_block_dropout_rate)(x)
            
            x = tf.keras.layers.Add()([x_, x])
            x_ = x
            x = tf.keras.layers.ReLU()(x)


        x = tf.keras.layers.Conv1D(filters[2], kernel_sizes[2], dilation_rate=dilation_rates[2], padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.Dropout(dropout_rates[2])(x)
        x = tf.keras.layers.Conv1D(filters[3], kernel_sizes[3], dilation_rate=dilation_rates[3], padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(dropout_rates[3])(x)

        x = tf.keras.layers.Conv1D(filters[4], kernel_sizes[4], dilation_rate=dilation_rates[4], padding='same', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(dropout_rates[4])(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(latent_dim, kernel_initializer='he_normal')(x)
        x = decoder(x)
        for k,v in decoder._get_trainable_state().items():
            k.trainable = False

        model = tf.keras.Model(inp, x)
        att_models = []
        for att in atts:
            att_models.append(tf.keras.Model(inp, att))
        self.input = inp
        self.output = x
        self.model = model
        self.attention_outputs = att_models
        

class VAE():
    def __init__(self,
                 data,
                 latent_dim=64):
        self.latent_dim = latent_dim
        h = data.y_val[0].shape[1]
        w = data.y_val[0].shape[2]

        inputs = tf.keras.Input(shape=(h, w, 1))
        x = tf.keras.layers.Conv2D(16, 5, activation="relu", strides=1, padding="same")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(latent_dim)(x)
        z_log_sigma = tf.keras.layers.Dense(latent_dim)(x)
        z = tf.keras.layers.Lambda(self.sampling)([z_mean, z_log_sigma])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        model_encoder = tf.keras.Model(inputs, z_mean, name='model_encoder')

        latent_inputs = tf.keras.Input(shape=(latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(h // 8 * w // 8 * 64, activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((h // 8, w // 8, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 1, activation="relu", strides=1, padding="same")(x)
        outputs = tf.keras.layers.Dense(1)(x)
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name='vae')
        
        reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
        reconstruction_loss *= h * w
        reconstruction_loss = K.mean(reconstruction_loss)
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = reconstruction_loss + kl_loss

        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        self.encoder = model_encoder
        self.decoder = decoder
        self.vae = vae
        

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                mean=0., stddev=0.05)
        return z_mean + K.exp(z_log_sigma) * epsilon


class Chimaera():
    def __init__(self,
                 data,
                 model_dir=None,
                 latent_dim = 64,
                 filters = [64, 128, 256, 512, 64],
                 kernel_sizes = [19, 9, 5, 3, 3],
                 dilation_rates = [1, 2, 16, 1, 1],
                 n_residual_blocks = 0,
                 n_attention_blocks = 0,
                 residual_block_filters = [256, 128],
                 residual_block_kernel_sizes = [5, 5],
                 residual_block_dilation_rate_factors = [4, 4],
                 querry_dim = 32,
                 n_heads = 8,
                 dropout_rates = [0, 0.1, 0.15, 0.15, 0.15],
                 residual_block_dropout_rates = [0, 0],
                 attention_block_dropout_rate = 0.1):
        try:
            custom_objects = {'SinePositionEncoding': keras_nlp.layers.SinePositionEncoding}
        except:
            custom_objects = {}
        self.model_1d = None
        self.data = data

        if model_dir is not None:
            availible_files = os.listdir(model_dir)
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'),
                                                    custom_objects=custom_objects)
            self.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'enc.h5'))
            self.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'dec.h5'))
            self.ae = None
            self.attention_outputs = []
            self.latent_model = None
        else:
            vae = VAE(data=data, latent_dim=latent_dim)
            self.encoder = vae.encoder
            self.decoder = vae.decoder
            self.ae = vae.vae

            main_model = MainModel(data = data,
                                   decoder = self.decoder,
                                    latent_dim = latent_dim,
                                    filters = filters,
                                    kernel_sizes = kernel_sizes,
                                    dilation_rates = dilation_rates,
                                    n_residual_blocks = n_residual_blocks,
                                    n_attention_blocks = n_attention_blocks,
                                    residual_block_filters = residual_block_filters,
                                    residual_block_kernel_sizes = residual_block_kernel_sizes,
                                    residual_block_dilation_rate_factors = residual_block_dilation_rate_factors,
                                    querry_dim = querry_dim,
                                    n_heads = n_heads,
                                    dropout_rates = dropout_rates,
                                    residual_block_dropout_rates = residual_block_dropout_rates,
                                    attention_block_dropout_rate = attention_block_dropout_rate)
            inp = main_model.input
            latent_output = main_model.output
            self.attention_outputs = main_model.attention_outputs
            self.latent_model = tf.keras.Model(inp, latent_output)
            self.model = self.latent_model
            self.model.compile(optimizer='adam', loss='mse')
