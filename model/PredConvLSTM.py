class Model():
    """
    model constructed
    @generator
    @model_load: loading model
    @_convlstm: convlstm 
    @encoder
    @
    """
    def generator(self, videos):
        '''
        This is the G in the model.
        @videos: [batch_size, 7, 256, 256, 3]
        return [batch_size, 32, 32, 512]
        '''
        #with tf.variable_scope("encoder") as scope:
        im_list = tf.unstack(videos,axis=1)

        z_enc = [self.encoder(im_list[i]) if i == 0 \
                else self.encoder(im_list[i], True)\
                for i in range(len(im_list))]

        self.z_enc = tf.stack(z_enc, axis = 0)

        #self.guessed_z = self.convlstm(self.z_enc)
        self.guessed_z = self._convlstm(self.z_enc)
        self.generated_images = self.decoder(self.guessed_z)
        self._summary_video(im_list)
        return self.generated_images
    
    
    def model_load(self, logdir, var_list = None):
        '''
        This is the model of saving and restore the whole graph.
        @var_list: a list for variables that restore previous model
        '''
        print(logdir)
        if not var_list:
            self.saver = tf.train.Saver(max_to_keep=2)
        else: 
            self.saver = tf.train.Saver(var_list,max_to_keep=2)
            
            print("partial")
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print("Model restored...")
            if var_list:
                print("partial")
    
    
    def _convlstm(self, z_enc):
        h = z_enc
        with tf.variable_scope("convlstm") as scope:
            cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2, input_shape=[32, 32, 512], output_channels=512, kernel_shape=[4, 4],name='conv_lstm_cell_init')
            initial_state = cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            for t in range(self.timesteps):
                output = h[t,:]

                if t > 0: 
                    ## 
                    tf.get_variable_scope().reuse_variables()
                    z_output,z_state =  cell(output,z_state)
                else:
                    z_output,z_state =  cell(output,initial_state)
            guessed_z = z_state.h
        return guessed_z
    
    
    def _summary_video(self, im_list):
        im_list.append(self.images)
        im_list.append(self.generated_images)
        im = tf.concat(im_list,axis=2)
        tf.summary.image("image",im)
    
    
    def tensorboard_summary(self):
        tf.summary.scalar("mse_loss", self.mse_loss)
        tf.summary.histogram("guessed_z",self.guessed_z)
        
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        
        
    def decoder(self, z):
        h=z
        shape = z.get_shape().as_list()
        with tf.variable_scope("decoder") as scope:
            N_dec = len(self.dec_dims)
            size = shape[1]
            for index in range(N_dec - 1):
                size *= 2
                W = utils.weight_variable([4, 4, self.dec_dims[-index-2],self.dec_dims[-index -1]], name="W_%d" % index)
                b = tf.zeros([self.dec_dims[-index - 2]])
                deconv_shape = tf.stack([tf.shape(h)[0], size, size, self.dec_dims[-index - 2]])
                h_conv_t = utils.conv2d_transpose_strided(h, W, b, output_shape=deconv_shape)
                #h_bn = batch_norm(h_conv_t, dims[index + 1], train_phase, scope="gen_bn%d" % index)
                h_bn = tf.contrib.layers.batch_norm(h_conv_t, decay = 0.9, epsilon=1e-5, scale=True, is_training=True, scope="disc_bn%d" % index)
                h = self.activation(h_bn, name='h_%d' % index)
                utils.add_activation_summary(h)

                

            W_pred = utils.weight_variable([5, 5, self.chanels, self.dec_dims[0]], name="W_pred")
            b = tf.zeros([self.chanels])
            deconv_shape = tf.stack([tf.shape(h)[0], size, size, self.chanels])

            h_conv_t = tf.nn.conv2d_transpose(h, W_pred, deconv_shape, strides=[1, 1, 1, 1], padding="SAME")
            tf.nn.bias_add(h_conv_t, b)
            #h_conv_t = utils.DisLossconv2d_transpose_strided(h, W_pred, b, output_shape=deconv_shape)
            pred_image = self.activation(h_conv_t, name='pred_image')
            utils.add_activation_summary(pred_image)
        return pred_image
    
    
    # encoder
    def encoder(self, input_images, scope_reuse = False):
        """
        @input_images: a batch of images as batch_size*length*width*chanels
        @enc_dims: a list for inputed dimenstions
        @
        """
        with tf.variable_scope("encoder") as scope:
            
            if scope_reuse:
                print("scope resue %s"%tf.get_variable_scope())
                tf.get_variable_scope().reuse_variables()
            
            N_enc = len(self.enc_dims)
            x_shape = input_images.get_shape().as_list()
            h = input_images

            W = utils.weight_variable(shape=[4,4,self.chanels,self.enc_dims[0]], name='W_init')
            b = utils.bias_variable([self.enc_dims[0]], name="b_init")
            conv = tf.nn.conv2d(h, W, strides=[1, 1, 1, 1], padding="SAME")
            h_conv = tf.nn.bias_add(conv, b)
            h = self.activation(h_conv,name = "h_init")
            if not scope_reuse: utils.add_activation_summary(h)
            
            # the loop for latent layers
            for index in range(N_enc-1):
                W = utils.weight_variable(shape = [4,4,self.enc_dims[index],self.enc_dims[index+1]],name="W_%d"%index)
                b = utils.bias_variable([self.enc_dims[index+1]], name="b_%d"%index)
                conv = utils.conv2d_strided(h,W,b)
                h_bn = tf.contrib.layers.batch_norm(conv, decay = 0.9, epsilon=1e-5, scale=True, is_training=True, scope="gen_bn%d"%index)
                h = self.activation(h_bn,name="h_%d"%index)
                if not scope_reuse: utils.add_activation_summary(h)
        return h