def images2spike(x, y, batch_size, shuffle, **kwargs):  
    '''Converts images to spike trains'''
    labels_ = np.array(y,dtype=np.int64)
    number_of_batches = len(x)//batch_size
    sample_index = np.arange(len(x))

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0

    batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
    while counter < number_of_batches:
        x_batch = torch.empty((len(x[batch_index]), nb_steps, nb_inputs)).to(device)
        for i, image in enumerate(x[batch_index]):
            tensor_image = torch.Tensor(image) # probabilities tensor
            scaled_image, average_rate = average_rate(tensor_image, scale, time_step)
            spike_train = torch.empty((nb_steps, nb_inputs))
            for t in range(nb_steps):
                spike_t = torch.bernoulli(tensor_image)
                spike_train[t] = spike_t
            x_batch[i] = spike_train
        y_batch = torch.tensor(labels_[batch_index],device=device) 

        yield x_batch,  y_batch

        counter += 1

def average_rate(image, scale, time_step):
    scaled_image = image*scale
    for i,prob in enumerate(scaled_image):
        if (prob > 1):
            new_prob = 1
            scaled_image[i] = new_prob
        elif (prob < 0):
            new_prob = 0
            scaled_image[i] = new_prob
        else:
            scaled_image[i] = prob
    rate_of_scaled_image = scaled_image/time_step
    average_rate = np.mean(rate_of_scaled_image)
    return scaled_image, average_rate
