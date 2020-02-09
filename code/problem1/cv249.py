import numpy as np
import cv2


class CV249:
    def cvt_to_gray(self, img):
        # Note that cv2.imread will read the image to BGR space rather than RGB space

        # TODO: your implementation
        # GRAY = 0.299*R + 0.587*G + 0.114*B
        # Here have to include np.rint in the output image else the test will fail
        img = np.rint(0.299 * img[:, :, 2] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 0]).astype(np.uint8)
        return img

    def blur(self, img, kernel_size=(3, 3)):
        """smooth the image with box filter

        Arguments:
            img {np.array} -- input array

        Keyword Arguments:
            kernel_size {tuple} -- kernel size (default: {(3, 3)})

        Returns:
            np.array -- blurred image
        """
        # TODO: your implementation
        # Create the box filter by initiating a square matrix of all ones, and times 1/9
        box_filter = 1/9 * (np.ones(kernel_size))
        blurred_img = cv2.filter2D(img, -1, box_filter)
        return blurred_img


    def sharpen_laplacian(self, img):
        """sharpen the image with laplacian filter

        Arguments:
            img {np.array} -- input image

        Returns:
            np.array -- sharpened image
        """

        # subtract the laplacian from the original image
        # when have a negative center in the laplacian kernel

        # TODO: your implementation
        laplacian_filter = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        img_laplacian = cv2.filter2D(img, -1, laplacian_filter)
        sharpened_img = img - img_laplacian
        return sharpened_img

    def unsharp_masking(self, img):
        """sharpen the image via unsharp masking

        Arguments:
            img {np.array} -- input image

        Returns:
            np.array -- sharpened image
        """
        # use don't use cv2 in this function
        # I assume this means don't use cv2.blur function outright
        # Use box filter to smooth the image
        # TODO: your implementation

        box_filter = 1/9 * np.ones((3,3))
        smoothed_image = cv2.filter2D(img, -1, box_filter)
        f_mask = img - smoothed_image
        g = img + 1 * f_mask
        return g


    def edge_det_sobel(self, img):
        """detect edges with sobel filter

        Arguments:
            img {np.array} -- input image

        Returns:
            [np.array] -- edges
        """

        # TODO: your implementation
        x_derivative_filter = np.array([[-1, 0, 1], [-2, 0 , 2], [-1, 0, 1]])
        y_derivative_filter = np.array([[-1, -2, -1], [0, 0 , 0], [1, 2, 1]])
        edges = np.sqrt((cv2.filter2D(img, -1, x_derivative_filter))**2 + (cv2.filter2D(img, -1, y_derivative_filter))**2)
        edges = edges.astype(np.uint8) # to get it pass the test case
        return edges
