# -*- coding: utf-8 -*-

import tensorflow as tf

def gen_conv(batch_input, out_channels, stride,ksize):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [ksize, ksize, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d(batch_input, filter, [1, stride, stride, 1], padding='SAME')
        return conv
    
def lrelu(x, a):
    with tf.name_scope("lrelu"):
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        input = tf.identity(input)
        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        variance_epsilon = 1e-5
        normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset
        return normalized, mean, variance

def deconv(batch_input, out_channels):
   with tf.variable_scope("deconv"):
        in_height, in_width, in_channels = [int(batch_input.shape[1]), int(batch_input.shape[2]), int(batch_input.shape[3])]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter1 = tf.get_variable("filter1", [4, 4, out_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        resized_images = tf.image.resize_images(batch_input, [in_height * 2, in_width * 2], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = tf.nn.conv2d(resized_images, filter, [1, 1, 1, 1], padding="SAME")
        conv = tf.nn.conv2d(conv, filter1, [1, 1, 1, 1], padding="SAME")
        return conv
    
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)  

#%%
def unetencoder(encoder_inputs):
    layers=[]

    with tf.variable_scope("conv_1"):
        convolved = gen_conv(encoder_inputs, 9 , stride=1, ksize=5)       
        layers.append(convolved)
    with tf.variable_scope("conv_2"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = gen_conv(rectified, 64 , stride=1, ksize=3)
        output,_,_ = instancenorm(convolved)
        layers.append(output)
    with tf.variable_scope("conv_3_down"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = gen_conv(rectified, 128 , stride=2, ksize=3)
        output,_,_ = instancenorm(convolved)
        layers.append(output)
    with tf.variable_scope("conv_4_down"):
        rectified = lrelu(layers[-1], 0.2)
        convolved = gen_conv(rectified, 256 , stride=2, ksize=3)
        output,_,_ = instancenorm(convolved)
        layers.append(output)   
    return layers[-1]

def decoder_nr(resout,outc,reuse=False):
    with tf.variable_scope("denr_") as scope:
        if reuse:
            scope.reuse_variables()
        layers = [resout,resout]
        with tf.variable_scope("conv_11"):
            convolved = gen_conv(layers[-1], 512 , stride=1, ksize=3)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_12_up"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = deconv(rectified, 256)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_13_up"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = deconv(rectified, 64)#
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_14"):
            rectified = lrelu(layers[-1], 0.2)
            output = deconv(rectified, outc)
#            output = gen_conv(rectified, outc, stride=1, ksize=3)
            output = tf.tanh(output)
            layers.append(output)   
    return layers[-1] 

def decoder_ds(resout,outc,reuse=False):
    with tf.variable_scope("deds_") as scope:
        if reuse:
            scope.reuse_variables()
        layers = [resout,resout]
        with tf.variable_scope("conv_11"):
            convolved = gen_conv(layers[-1], 512 , stride=1, ksize=3)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_12_up"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = deconv(rectified, 256)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_13_up"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = deconv(rectified, 64)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv_14"):
            rectified = lrelu(layers[-1], 0.2)
            output = deconv(rectified, outc)
#            output = gen_conv(rectified, outc, stride=1, ksize=3)
            output = tf.tanh(output)
            layers.append(output)   
    return layers[-1]

def latentz_encoder(inputs,reuse=False):
    with tf.variable_scope("en_") as scope:
        if reuse:
            scope.reuse_variables()
        latentz = unetencoder(inputs)
    return latentz

def height_to_normal(height): 
    # input: height map, [bs, w, h, 1], (0, 1)
    # output: normal map, [bs, w, h, 3], normalized, (-1, 1)    
    c1 = 32
    c2 = 32
    dx = height[:, 1:, :, :] - height[:, :-1, :, :]
    dx_zeros_shape = (height.shape[0],1,height.shape[2],height.shape[3])
    dx_zeros = tf.zeros(dx_zeros_shape)
    dx = tf.concat([dx,dx_zeros],axis=1)
    
    dy = height[:, :, 1:, :] - height[:, :, :-1, :]
    dy_zeros_shape = (height.shape[0],height.shape[1],1,height.shape[3])
    dy_zeros = tf.zeros(dy_zeros_shape)
    dy = tf.concat([dy,dy_zeros],axis=2)
    
    # dx, dy = tf.image.image_gradients(height)
    
    ddx = c1 * dy 
    ddy = c2 * dx 
    one = tf.ones_like(ddx)
    n = tf.concat([-ddx, -ddy, one], axis=-1)
    n = tf_Normalize(n)
    return n

def generator(latentz,reuse_bool = False):
    OutputedHR = decoder_nr(latentz,outc=2,reuse=reuse_bool)
    OutputedDS = decoder_ds(latentz,outc=4,reuse=reuse_bool)
    partialOutputedheight = OutputedHR[:,:,:,0:1]
    outputedRoughness = OutputedHR[:,:,:,1]
    outputedDiffuse = OutputedDS[:,:,:,0:3] 
    outputedSpecular = OutputedDS[:,:,:,3] 

    normNormals = height_to_normal((partialOutputedheight+1)/2)
    
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    outputedRoughnessMap = tf.concat([outputedRoughnessExpanded,outputedRoughnessExpanded,outputedRoughnessExpanded],axis=-1)
    outputedSpecularExpanded = tf.expand_dims(outputedSpecular, axis = -1)
    outputedSpecularMap = tf.concat([outputedSpecularExpanded,outputedSpecularExpanded,outputedSpecularExpanded],axis=-1)    
    reconstructedOutputs =  tf.concat([normNormals, outputedDiffuse, outputedRoughnessMap, outputedSpecularMap], axis=-1)
    
    return reconstructedOutputs

#%%
def sigmoid_cross_entropy_with_logits(x, y):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

def Discriminator_patch(generator_inputs,reuse=False):
    with tf.variable_scope("d_discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        layers=[]
        with tf.variable_scope("conv1"):
            convolved = gen_conv(generator_inputs, 64 , stride=2,ksize=4)        
            layers.append(convolved)
        with tf.variable_scope("conv2"):            
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, 128 , stride=2, ksize=4)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv3"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, 256 , stride=2, ksize=4)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
        with tf.variable_scope("conv4"):
            rectified = lrelu(layers[-1], 0.2)
            convolved = gen_conv(rectified, 512 , stride=2, ksize=4)
            output,_,_ = instancenorm(convolved)
            layers.append(output)
             
        rectified = lrelu(layers[-1], 0.2)
        convolved = gen_conv(rectified, out_channels=1, stride=1, ksize=4)
        output = convolved
        
        return output  
    
def patchGAN_d_loss(disc_fake,disc_real):
    loss_d_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
    loss_d_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
    disc_cost = loss_d_real + loss_d_fake
    return disc_cost

def patchGAN_g_loss(disc_fake):
    gen_cost = tf.reduce_mean(sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))    
    return gen_cost

def CTRender(M, eview_vec, elight_vec):
    # M.get_shape() [1,2x,2x,12]
    half_vec = (eview_vec + elight_vec) / 2
    half_vec_norm = tf.sqrt(tf.reduce_sum(tf.square(half_vec),axis=-1))
    half_vec_expand = tf.expand_dims(half_vec_norm,axis=-1)
    newhalf_vec = tf.concat([half_vec_expand,half_vec_expand,half_vec_expand],axis=-1)
    ehalf_vec = half_vec/newhalf_vec
    
    norm, diff, rough, spec = tf.split(M, 4, axis=3)
    norm = norm*2-1 
    norm_mo = tf.sqrt(tf.reduce_sum(tf.square(norm),axis=-1))
    norm_expand = tf.expand_dims(norm_mo,axis=-1)
    newnorm = tf.concat([norm_expand,norm_expand,norm_expand],axis=-1)
    enorm = norm/newnorm
    
    pi = tf.constant(3.1415926)
    diff_scale = diff/pi
    
    NdotH = tf.reduce_sum(tf.multiply(enorm,ehalf_vec),axis=-1)
    nh_expand = tf.expand_dims(NdotH,axis=-1)
    nh = tf.concat([nh_expand,nh_expand,nh_expand],axis=-1)

    NdotL = tf.reduce_sum(tf.multiply(enorm,elight_vec),axis=-1)
    nl_expand = tf.expand_dims(NdotL,axis=-1)
    nl = tf.concat([nl_expand,nl_expand,nl_expand],axis=-1)

    NdotV = tf.reduce_sum(tf.multiply(enorm,eview_vec),axis=-1)
    nv_expand = tf.expand_dims(NdotV,axis=-1)
    nv = tf.concat([nv_expand,nv_expand,nv_expand],axis=-1)    
    
    VdotH = tf.reduce_sum(tf.multiply(eview_vec,ehalf_vec),axis=-1)
    vh_expand = tf.expand_dims(VdotH,axis=-1)
    vh = tf.concat([vh_expand,vh_expand,vh_expand],axis=-1)
    
    nh = tf.maximum(nh, 1e-8)
    nv = tf.maximum(nv, 1e-8)
    nl = tf.maximum(nl, 1e-8)
    vh = tf.maximum(vh, 1e-8)
    
    r2 = rough*rough
    denominator = tf.square(nh)*(tf.pow(r2,2)-1)+1+0.0001
    Norm_distrib = r2/pi/ denominator
    norm_distrib = tf.square(Norm_distrib)*pi

    k = tf.maximum(1e-8, rough * rough * 0.5)
    shade_mask1 = nl*(1-k)+k
    shade_mask2 = nv*(1-k)+k
    shade_mask = tf.reciprocal(shade_mask1*shade_mask2)
    
    F_mi = (-5.55473*vh-6.98316)*vh
    F_mi = tf.cast(F_mi,tf.float32)
    fresnel = spec+(1-spec)*tf.pow(2.0,F_mi)
    
    fr = fresnel*shade_mask*norm_distrib/4 + diff_scale

    ctrender_batch = fr*nl*3.14

    return ctrender_batch

#%%
def generate_vl(w=3264, h=2448):

    # d = w/(2*tf.tan(alpha)), alpha = 33/180*pi
    d = w/1.29876
    view_pos = tf.constant([w/2, h/2, d], dtype=tf.float32, shape=[1,3])
    view_pos = tf.expand_dims(view_pos,axis=1)
    
    wgrid = tf.linspace(0.0, w, w)
    hgrid = tf.linspace(0.0, h, h)
    plane_coor = tf.concat(tf.meshgrid(wgrid,hgrid,0.0),axis=2)
    planc_expand = tf.expand_dims(plane_coor,axis=0) #[1, h, w, 3]
    
    xy = planc_expand[:,:,:,0:2]
    t = tf.sqrt(tf.reduce_sum(tf.square(xy - tf.constant([w/2, h/2], dtype=tf.float32, shape=[1,2])), keep_dims=True, axis=-1))
    I = tf.exp(tf.square(tf.tan(t/d*1.6)) * (-0.5))
    I = tf.concat([I,I,I], axis=-1)
    
    view_pos_expand = tf.expand_dims(view_pos,axis=1)
    view_vec = view_pos_expand - planc_expand
    
    view_norm = tf.sqrt(tf.reduce_sum(tf.square(view_vec),axis=-1))
    view_expand = tf.expand_dims(view_norm,axis=-1)
    newview_vec = tf.concat([view_expand,view_expand,view_expand],axis=-1)
    eview_vec = view_vec/newview_vec
    
    return eview_vec, I