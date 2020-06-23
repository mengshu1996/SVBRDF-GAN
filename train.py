# -*- coding: utf-8 -*-

import tensorflow as tf
import net
import os
import collections
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True)
parser.add_argument("--log_dir", required=True)
parser.add_argument("--output_dir", required=True)

parser.add_argument("--crop_size", type = int, default=800)
parser.add_argument("--img_h", type = int, default=1224)
parser.add_argument("--img_w", type = int, default=1632)
parser.add_argument("--max_step", type = int, default=20000)
parser.add_argument("--seed", type = int, default=24)

args,unknown = parser.parse_known_args()

tf.set_random_seed(args.seed)
np.random.seed(args.seed)

ganscale=0.1
nbTargets=4
inputsize=128
BATCH_SIZE = 1
Examples = collections.namedtuple("Examples", "iterator, concats")
lr=0.00002
#%%
def deprocess(image):
    # [-1, 1] => [0, 1]
    with tf.name_scope("deprocess"):
        return (image + 1) / 2

def crop_imgs(raw_input,tile_size):
    concat_tile = tf.random_crop(raw_input, size=[tile_size*2,tile_size*2,12],seed=args.seed)
    return concat_tile

#%%
def gaussian_kernel(size=5,sigma=2):
    x_points = np.arange(-(size-1)//2,(size-1)//2+1,1)
    y_points = x_points[::-1]
    xs,ys = np.meshgrid(x_points,y_points)
    kernel = np.exp(-(xs**2+ys**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    return kernel/kernel.sum()

def scale(imgtile):
    # =>[0,1]
    minpixel = tf.reduce_min(imgtile,axis=[1,2],keep_dims=True)
    maxpixel = tf.reduce_max(imgtile,axis=[1,2],keep_dims=True)
    scaleimg = (imgtile - minpixel)/(maxpixel - minpixel+0.0001)
    return scaleimg

def blur(x,kernel):
    kernel = kernel[:, :, np.newaxis, np.newaxis]
    xr,xg,xb =tf.expand_dims(x[:,:,:,0],-1),tf.expand_dims(x[:,:,:,1],-1),tf.expand_dims(x[:,:,:,2],-1)
    xr_blur = tf.nn.conv2d(xr, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xg_blur = tf.nn.conv2d(xg, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xb_blur = tf.nn.conv2d(xb, kernel, strides=[1, 1, 1, 1], padding='SAME')
    xrgb_blur = tf.concat([xr_blur,xg_blur,xb_blur],axis=3)   
    return xrgb_blur

def normalize_aittala(input_img,kernel1):     
    y_mean = blur(input_img,kernel1)
    y_stddv = tf.sqrt(blur(tf.square(input_img - y_mean),kernel1))
    norm = (input_img - y_mean)/(y_stddv+0.0001)
    scaley = scale(norm)
    return scaley

def concat_inputs(filename,kernel1):
    flashimg_string = tf.read_file(filename)
    flash_input = tf.image.decode_image(flashimg_string)
    flash_input = tf.image.convert_image_dtype(flash_input, dtype=tf.float32)
    flash_input = tf.expand_dims(flash_input**2.2,axis=0)

    initdiffuse = normalize_aittala(flash_input,kernel1)
    wv,inten=net.generate_vl(args.img_w,args.img_h)
    img_tobe_sliced = tf.concat([flash_input,wv,inten,initdiffuse],axis=-1)
    return img_tobe_sliced

def load_examples(img_tobe_sliced,tilesize=256):
    dataset = tf.data.Dataset.from_tensor_slices(img_tobe_sliced)
    dataset = dataset.map(lambda x:crop_imgs(x,tile_size=tilesize))#, num_parallel_calls=1)
    dataset = dataset.repeat()
    batched_dataset = dataset.batch(BATCH_SIZE)#.prefetch(tf.contrib.data.AUTOTUNE)
    iterator = batched_dataset.make_initializable_iterator()
    concat_batch = iterator.get_next()
    return Examples(
        iterator=iterator,
        concats=concat_batch,
    )

def save_outputs(predictions,examples_inputs=None,net_rerender=None):
    
    n,d,r,s = tf.split(predictions, nbTargets, axis=3)#4 * [batch, 256,256,3]
    gammad = d**(1/2.2)
    outputs_list=[n,gammad,r,s]
    outputs = tf.stack(outputs_list, axis = 1) #[batch, 4,256,256,3]
    shape = tf.shape(outputs)
    newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
    outputs_reshaped = tf.reshape(outputs, newShape)
    converted_outputs = tf.image.convert_image_dtype(outputs_reshaped, dtype=tf.uint16, saturate=True)

    input_batch = tf.image.convert_image_dtype(examples_inputs**(1/2.2), dtype=tf.uint16, saturate=True)
    rerender_batch = tf.image.convert_image_dtype(net_rerender**(1/2.2), dtype=tf.uint16, saturate=True)
    display_fetches = {
    "inputs": tf.map_fn(tf.image.encode_png, input_batch, dtype=tf.string, name="input_pngs"),        
    "rerenders": tf.map_fn(tf.image.encode_png, rerender_batch, dtype=tf.string, name="rerender_pngs"),
    "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),   
    }
    return display_fetches

def save_images(fetches, output_dir, step=None, mode="images"):
    image_dir = os.path.join(output_dir, mode)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i in range(BATCH_SIZE):

        fileset = {"step": step}
        #fetch inputs
        kind = "inputs"
        filename = "batch"+ str(i)+"-" + kind +".png"
        if step is not None:
            filename = "%05d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        #fetch rerenders
        kind = "rerenders"
        filename = "batch"+ str(i)+"-" + kind +".png"
        if step is not None:
            filename = "%05d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        with open(out_path, "wb") as f:
            f.write(contents) 

        #fetch outputs
        for kind in ["outputs"]:
            for idImage in range(nbTargets):
                filename = "batch"+ str(i)+"-"+ kind + "-" + str(idImage) + "-.png"
                if step is not None:
                    filename = "%05d-%s" % (step, filename)
                filetsetKey = kind + str(idImage)
                fileset[filetsetKey] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i * nbTargets + idImage]
                with open(out_path, "wb") as f:
                    f.write(contents)
                    
        filesets.append(fileset)
    return filesets
#%%
def predict():
    cropsize = args.crop_size
    img_string = tf.read_file(args.input_dir)
    flash_input = tf.image.decode_image(img_string)
    flash_input = tf.image.convert_image_dtype(flash_input, dtype=tf.float32)
    begin = (int((args.img_h-cropsize)/2),int((args.img_w-cropsize)/2),0)
    flash_crop = tf.slice(flash_input,begin,[cropsize,cropsize,3])
    flash_crop = tf.reshape(tf.expand_dims(flash_crop**2.2,axis=0),[1,cropsize,cropsize,3])    
    latentcode = net.latentz_encoder(flash_crop,True)
    predictions = deprocess(net.generator(latentcode,True)) 
    wv,inten = net.generate_vl(cropsize*2,cropsize*2)
    rerender = net.CTRender(predictions,wv,wv)
    display_fetches = save_outputs(predictions,flash_crop,rerender)
    return display_fetches    

def main():
    kernel1 = gaussian_kernel(101,101)
    input_2b_sliced = concat_inputs(args.input_dir,kernel1)
    examples = load_examples(input_2b_sliced,inputsize)    
    
    examples_flashes = examples.concats[:,:,:,0:3]
    examples_inputs = tf.map_fn(lambda x:tf.image.central_crop(x, 0.5),elems=examples_flashes)
    examples_inputs = tf.reshape(examples_inputs,[BATCH_SIZE,inputsize,inputsize,3])
    wv = examples.concats[:,:,:,3:6]
#    inten = examples_concats[:,:,:,6:9]
    initd = examples.concats[:,:,:,9:12]
    
    latentcode=net.latentz_encoder(examples_inputs)
    predictions = deprocess(net.generator(latentcode))
    net_rerender = net.CTRender(predictions,wv,wv)
        
    prediffuse = predictions[:,:,:,3:6]
    
    dis_real = net.Discriminator_patch(examples_flashes,reuse=False)
    dis_fake = net.Discriminator_patch(net_rerender,reuse=True)
    dis_cost = net.patchGAN_d_loss(dis_fake,dis_real)
    gen_fake = net.patchGAN_g_loss(dis_fake)
    
    train_vars = tf.trainable_variables() 
#    encoder_vars = [v for v in train_vars if 'en_' in v.name]
    decodernr_vars = [v for v in train_vars if 'denr_' in v.name]
    decoderds_vars = [v for v in train_vars if 'deds_' in v.name]
    discriminator_vars = [v for v in train_vars if 'd_' in v.name]
    
    gnr_vars = decodernr_vars
    gds_vars = decoderds_vars
    
    diffuseloss = tf.reduce_mean(tf.abs(prediffuse - initd))
    gnr_cost = ganscale*(gen_fake) + diffuseloss
    gds_cost = ganscale*(gen_fake) + diffuseloss
    
    gnr_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='gnr_opt',
    				beta1=0., beta2=0.9).minimize(gnr_cost, var_list=gnr_vars)
    gds_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='gds_opt',
    				beta1=0., beta2=0.9).minimize(gds_cost, var_list=gds_vars)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='d_opt',
     				beta1=0., beta2=0.9).minimize(dis_cost, var_list=discriminator_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    g_loss_summary = tf.summary.scalar("g_loss", gnr_cost)
    d_loss_summary = tf.summary.scalar("d_loss", dis_cost)
    train_summary = tf.summary.merge([g_loss_summary,d_loss_summary])
    train_writer = tf.summary.FileWriter(args.log_dir, sess.graph)
     
    saver = tf.train.Saver(max_to_keep=10)
    sess.run([examples.iterator.initializer,tf.global_variables_initializer()])
     
    for step in range(args.max_step):
    
        for i in range(5):        
            _, g_loss = sess.run([gnr_optimizer, gnr_cost])
        for i in range(1):
            _, g_loss = sess.run([gds_optimizer, gds_cost])
            
        _, d_loss = sess.run([d_optimizer, dis_cost]) 
        
        if step % 500 == 0 or (step + 1) == args.max_step:
            print('Step %d,  g_loss = %.4f, d_loss = %.4f' %(step, g_loss, d_loss))
            summary_str = sess.run(train_summary)         
            train_writer.add_summary(summary_str, step)
        if step % 1000 == 0 or (step + 1) == args.max_step:
            checkpoint_path = os.path.join(args.log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
            display_fetches = save_outputs(predictions,examples_flashes,net_rerender)
            results = sess.run(display_fetches)
            save_images(results, args.output_dir, step, "training_tiles")

            prediction_fetches = predict()
            predict_maps = sess.run(prediction_fetches)
            save_images(predict_maps, args.output_dir, step, "predicted_maps")

if __name__ == '__main__':
    main()

