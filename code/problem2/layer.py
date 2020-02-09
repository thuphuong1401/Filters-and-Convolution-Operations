import numpy as np

# NOTE: output channel is the number of filter one uses to convolve over the original image
class Conv2D(object):
    """2D convolutional layer.

    Arguments:
        kernel_size (tuple): the shape of the kernel. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (int or tuple): the strides of the convolution operation.
            padding (int or tuple): number of zero paddings.
        weights_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        bias_init (obj):  an object instantiated using any initializer class
                in the "initializer" module.
        name (str): the name of the layer.

    Attributes:
        W (np.array): the weights of the layer. A 4D array of shape (
            out_channels, in_channels, kernel_height, kernel_width).
        b (np.array): the biases of the layer. A 1D array of shape (
            out_channels).
        kernel_size (tuple): the shape of the filter. It is a tuple = (
            out_channels, in_channels, kernel_height, kernel_width).
        strides (tuple): the strides of the convolution operation. A tuple = (
            height_stride, width_stride).
        padding (tuple): the number of zero paddings along the height and
            width. A tuple = (height_padding, width_padding).
        name (str): the name of the layer.

    """

    def __init__(
            self, kernel_size, stride, padding):
        self.W = np.random.randn(*kernel_size)
        self.b = np.random.randn(kernel_size[0], 1)
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding

    def __repr__(self):
        return "{}({}, {}, {})".format(
            self.name, self.kernel_size, self.stride, self.padding
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Args:
            x (np.array): the input of the layer. A 3D array of shape (
                in_channels, in_heights, in_weights).

        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).

        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.W.shape[2] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.W.shape[3] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        # Parameters

        # Input
        img_height = x.shape[1]
        img_width = x.shape[2]

        # Output and kernel
        W = self.W # (out_channels, in_channels, kernel_height, kernel_width)
        out_channels = W.shape[0]
        in_channels = W.shape[1]
        kernel_height = W.shape[2]
        kernel_width = W.shape[3]
        bias = self.b

        # The default of stride_height and stride_width is 1
        stride_height = s[0] if s[0] != 0 else 1
        stride_width = s[1] if s[1] != 0 else 1

        # Ouput dimensions
        output_height = (img_height - kernel_height + 2*p[0]) // s[0] + 1
        output_width = (img_width - kernel_width + 2*p[1]) // s[1] + 1

        # Initialize empty output volume
        # // instead of / because in the fixed formula we are using the floor division function
        output = np.zeros((out_channels, output_height, output_width))


        # Convolve the image by each kernel
        for kernel_number in range(out_channels):
            # Obtain the current kernel
            curr_kernel = W[kernel_number, :] #in_channels, kernel_height, kernel_width

            #print(x_padded.shape)
            #print(curr_kernel.shape)
            #print(W.shape)
            #print(out_channels)


            # Idea: doing convolution over volumes
            # Loop through each row and column in the output
            for row in range(output.shape[1]):
                for col in range(output.shape[2]):
                    # Get starting row/col and ending row/col of the 'receptive field'
                    starting_row = row*stride_height
                    ending_row = row*stride_height + kernel_height
                    starting_col = col*stride_width
                    ending_col = col*stride_width + kernel_width

                    # The receptive field where convolution happens
                    # Different from the receptive field of pooling operation, the channel of the receptive field is the channel of the input.
                    receptive_field = x_padded[:, starting_row : ending_row, starting_col : ending_col]

                    # Do dot product of the receptive field with kernel
                    output[kernel_number, row, col] = np.sum(receptive_field * curr_kernel) + bias[kernel_number]

        return output


# NOTE: output channel same as input channel
class MaxPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        # Input
        img_height = x.shape[1]
        img_width = x.shape[2]
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]
        # Number of in_channels = number of out_channels
        out_channels = x.shape[0]

        # The default of stride_height and stride_width is 1
        stride_height = s[0] if s[0] != 0 else 1
        stride_width = s[1] if s[1] != 0 else 1

        # Ouput dimensions
        output_height = (img_height - kernel_height + 2*p[0]) // s[0] + 1
        output_width = (img_width - kernel_width + 2*p[1]) // s[1] + 1

        # Initialize empty output volume
        # // instead of / because in the fixed formula we are using the floor division function
        output = np.zeros((out_channels, output_height, output_width))


        # Loop through each channel
        for channel in range(out_channels): # or in_channels, they are equal
            # Loop through each row and column in the output
            for row in range(output.shape[1]):
                for col in range(output.shape[2]):
                    # Get starting row/col and ending row/col of the 'receptive field'
                    starting_row = row*stride_height
                    ending_row = row*stride_height + kernel_height
                    starting_col = col*stride_width
                    ending_col = col*stride_width + kernel_width

                    # The receptive field where pooling happens
                    receptive_field = x_padded[channel, starting_row : ending_row, starting_col : ending_col]

                    # Take max of numbers within the receptive field, then record it in the respective position in the output volume
                    output[channel, row, col] = np.max(receptive_field)

        return output

# NOTE: output channel same as input channel
class AvgPool2D:
    def __init__(self, kernel_size, stride, padding, name="MaxPool2D"):
        self.kernel_size = kernel_size
        self.stride = (stride, stride) if type(stride) == int else stride
        self.padding = (padding, padding) if type(padding) == int else padding
        self.name = name

    def __repr__(self):
        return "{}({}, {}, {})".format(
        self.name, self.kernel_size, self.stride, self.padding
    )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """Compute the layer output.

        Arguments:
            x {[np.array]} -- the input of the layer. A 3D array of shape (
                              in_channels, in_heights, in_weights).
        Returns:
            The output of the layer. A 3D array of shape (out_channels,
                out_heights, out_weights).
        """
        p, s = self.padding, self.stride
        x_padded = np.pad(
            x, ((0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant'
        )

        # check dimensions
        assert (x.shape[1] - self.kernel_size[0] + 2 * p[0]) / s[0] + 1 > 0, \
                'Height doesn\'t work'
        assert (x.shape[2] - self.kernel_size[1] + 2 * p[1]) / s[1] + 1 > 0, \
                'Width doesn\'t work'

        # TODO: Put your code below
        # Input
        img_height = x.shape[1]
        img_width = x.shape[2]
        kernel_height = self.kernel_size[0]
        kernel_width = self.kernel_size[1]
        # Number of in_channels = number of out_channels
        out_channels = x.shape[0]

        # The default of stride_height and stride_width is 1
        stride_height = s[0] if s[0] != 0 else 1
        stride_width = s[1] if s[1] != 0 else 1

        # Ouput dimensions
        output_height = (img_height - kernel_height + 2*p[0]) // s[0] + 1
        output_width = (img_width - kernel_width + 2*p[1]) // s[1] + 1

        # Initialize empty output volume
        # // instead of / because in the fixed formula we are using the floor division function
        output = np.zeros((out_channels, output_height, output_width))


        # Loop through each channel
        for channel in range(out_channels): # or in_channels, they are equal
            # Loop through each row and column in the output
            for row in range(output.shape[1]):
                for col in range(output.shape[2]):
                    # Get starting row/col and ending row/col of the 'receptive field'
                    starting_row = row*stride_height
                    ending_row = row*stride_height + kernel_height
                    starting_col = col*stride_width
                    ending_col = col*stride_width + kernel_width

                    # The receptive field where convolution happens
                    receptive_field = x_padded[channel, starting_row : ending_row, starting_col : ending_col]

                    # Take the mean of numbers within the receptive field and record in the respective position in the output volume
                    output[channel, row, col] = np.mean(receptive_field)

        return output
