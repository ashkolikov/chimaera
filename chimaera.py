#@title chimaera.chimaera

import tensorflow as tf
import tensorflow.keras.backend as K

class Chimaera():
    def __init__(self, data, latent_dim=64, net_type='simple', model_dir=None):
        self.latent_dim = latent_dim
        self.dna_len = data.dna_len
        self.data=data
        self.h = data.y_val[0].shape[1]
        self.w = data.y_val[0].shape[2]
        
        self.model_1d = None
        if model_dir is not None:
            availible_files = os.listdir(model_dir)
            self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
            self.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'enc.h5'))
            self.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'dec.h5'))
            self.ae = None
        else:   
            if net_type == 'simple':
                self.model = self.simple_model()
            elif net_type == 'resnet':
                self.model = self.resnet_model()
            elif net_type == 'complex':
                self.model = self.complex_model()
            elif net_type == 'attention':
                self.model = self.attention_model()
            self.encoder, self.decoder, self.ae = self.autoencoder()

            
    def complex_model(self):
        input = tf.keras.layers.Input(shape=(self.dna_len, 4))
        forward = input
        revcomp = tf.keras.layers.Lambda(lambda x: K.reverse(x, axes=(1, 2)))(forward)

        stack = tf.keras.layers.Concatenate(axis=1)([forward, revcomp])
        first_block = [tf.keras.layers.Conv1D(64, 15, padding='same', use_bias=False, 
                                   kernel_initializer='he_normal'),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.Dropout(0.1),
                       tf.keras.layers.ReLU(),
                       tf.keras.layers.MaxPooling1D(pool_size=2),
                       tf.keras.layers.Conv1D(64, 7, padding='same', use_bias=False, 
                                   kernel_initializer='he_normal'),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.Dropout(0.1),
                       tf.keras.layers.ReLU(),
                       tf.keras.layers.MaxPooling1D(pool_size=2),
                       tf.keras.layers.Conv1D(64, 5, padding='same', use_bias=False, 
                                   kernel_initializer='he_normal'),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.Dropout(0.1),
                       tf.keras.layers.ReLU(),
                       tf.keras.layers.MaxPooling1D(pool_size=2, name='aaaa')]
        
        for layer in first_block:
            stack = layer(stack)
            #revcomp = layer(revcomp)
        
        a = stack.shape[1]//2
        x = tf.keras.layers.Lambda(lambda x: K.concatenate([x[:,:a],
                                                            K.reverse(x[:,a:],
                                                                      axes=1)],
                                                            axis=2))(stack)

        #revcomp = tf.keras.layers.Lambda(lambda x: K.reverse(x, axes=1))(revcomp)
        #x = tf.keras.layers.Concatenate(axis=2)([forward, revcomp])

        pooling_range = int(np.log2(self.dna_len)) - 13
        #shortcuts = []
        for block in range(pooling_range):
            #shortcuts.append(x)
            x = tf.keras.layers.Conv1D(64, 5, padding='same',
                                       use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2, name='bbbbb')(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        
        '''for block in range(8):
            x = tf.keras.layers.Conv1D(32, 3, padding='same', dilation_rate=2**block,
                                       use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Dropout(0.1)(x)'''
        
        x = tf.keras.layers.Conv1D(64, 5, padding='same',
                                       use_bias=False, kernel_initializer='he_normal')(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv1D(64, 3, padding='same',
                                       use_bias=False, kernel_initializer='he_normal')(x)
        
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        output = tf.keras.layers.Dense(self.latent_dim, kernel_initializer='he_normal')(x)

        model = tf.keras.Model(input, output)

        model.compile(loss = 'mse', optimizer = 'adam')
        return model




    def simple_model(self):

        he = tf.keras.initializers.HeNormal()
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.dna_len, 4)))

        model.add(tf.keras.layers.Conv1D(64, 15, padding='same', dilation_rate = 1, activation='relu', use_bias=False, kernel_initializer='he_normal'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        pooling_range = int(np.log2(self.dna_len)) - 8
        for block in range(2):
            model.add(tf.keras.layers.Conv1D(32, 9, padding='same', dilation_rate = 2, activation='relu', use_bias=False, kernel_initializer='he_normal'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            model.add(tf.keras.layers.Dropout(0.15))
        
        for block in range(pooling_range-2):
            model.add(tf.keras.layers.Conv1D(16, 5, padding='same', dilation_rate = 4, activation='relu', use_bias=False, kernel_initializer='he_normal'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
            model.add(tf.keras.layers.Dropout(0.15))
        
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(self.latent_dim, kernel_initializer='he_normal'))

        opt = tf.keras.optimizers.Adam()
        model.compile(loss = 'mse', optimizer = opt)

        return model

    def resnet_model(self):
        l2 = tf.keras.regularizers.l2

        #he = tf.keras.initializers.HeNormal()
        input = tf.keras.layers.Input(shape=(self.dna_len, 4))

        x = tf.keras.layers.Conv1D(64, 15, padding='same', use_bias=False, 
                                   kernel_initializer='he_normal')(input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        

        pooling_range = int(np.log2(self.dna_len)) - 8
        for block in range(5):
            x = tf.keras.layers.Conv1D(64, 9, padding='same', use_bias=False,\
                                       kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Dropout(0.1)(x)

        x1 = x
        for block in range(5):         
            x = tf.keras.layers.Conv1D(64, 5, padding='same', dilation_rate = 2, use_bias=False, 
                                       kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.Conv1D(64, 5, padding='same', dilation_rate = 4, use_bias=False,
                                       kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)

            x = tf.keras.layers.Add()([x1, x])
            x = tf.keras.layers.Dropout(0.1)(x)
            
            x1 = tf.keras.layers.ReLU()(x)

        x = x1
        for block in range(pooling_range-5):
            x = tf.keras.layers.Conv1D(72, 5, padding='same', dilation_rate = 2,
                                       use_bias=False, kernel_initializer='he_normal')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
            x = tf.keras.layers.Dropout(0.15)(x)
        
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        output = tf.keras.layers.Dense(self.latent_dim, kernel_initializer='he_normal')(x)

        model = tf.keras.Model(input, output)

        model.compile(loss = 'mse', optimizer = 'adam')

        return model
    

    def attention_model(self):
        pass
    
        
    def sampling(self, args):

        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim),
                                mean=0., stddev=0.05)
        return z_mean + K.exp(z_log_sigma) * epsilon


    def autoencoder(self):

        inputs = tf.keras.Input(shape=(self.h, self.w, 1))
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
        z_mean = tf.keras.layers.Dense(self.latent_dim)(x)
        z_log_sigma = tf.keras.layers.Dense(self.latent_dim)(x)
        z = tf.keras.layers.Lambda(self.sampling)([z_mean, z_log_sigma])
        encoder = tf.keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
        model_encoder = tf.keras.Model(inputs, z_mean, name='model_encoder')

        latent_inputs = tf.keras.Input(shape=(self.latent_dim,), name='z_sampling')
        x = tf.keras.layers.Dense(self.h // 8 * self.w // 8 * 64, activation="relu")(latent_inputs)
        x = tf.keras.layers.Reshape((self.h // 8, self.w // 8, 64))(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(32, 1, activation="relu", strides=1, padding="same")(x)
        if self.data.normalize == 'standart':
            outputs = tf.keras.layers.Dense(1)(x)
        else:
            outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')

        outputs = decoder(encoder(inputs)[2])
        vae = tf.keras.Model(inputs, outputs, name='vae')

        
        if self.data.normalize == 'standart':
            reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
        else:
            reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.h * self.w
        reconstruction_loss = K.mean(reconstruction_loss)
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = reconstruction_loss + kl_loss

        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')

        return model_encoder, decoder, vae
        
