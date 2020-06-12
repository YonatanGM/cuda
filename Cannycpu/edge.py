import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    kernel = np.flipud(kernel)
    kernel = np.fliplr(kernel)
    out = np.copy(image)
    
    for i in range(pad_width0, Hi + pad_width0):
        for j in range(pad_width1, Wi + pad_width1):
            matrix = padded[i - pad_width0:i + pad_width0 + 1, j - pad_width1:j + pad_width1 + 1]
            out[i-pad_width0, j -pad_width1] = np.sum(matrix * kernel)

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            kernel[i][j] = (1/(2*np.pi*sigma**2.0)) * np.exp(((i-sigma)**2.0 + (j-sigma)**2.0)/(-2.0*sigma**2))

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """
    

    out = conv(img, np.array([[0.5, 0, -0.5]]))
    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = conv(img, np.array([[0.5, 0, -0.5]]).T)

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    
    H, W = img.shape
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    
    partialx = partial_x(img)
    partialy = partial_y(img)
    G = np.array([[np.sqrt(partialx[i][j]**2 + partialy[i][j]**2) for j in range(W)] for i in range(H)])
      
    theta = np.array([[np.arctan2(partialy[i][j],partialx[i][j]) for j in range(W)] for i in range(H)])

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta+22.5)/ 45.0) * 45

    for i in range(1, H-1):
        for j in range(1, W-1):
            if theta[i][j] == 0:
                if G[i][j] >= G[i][j-1] and G[i][j] >= G[i][j+1]: #0 
                    out[i][j] = G[i][j]
                 
            if theta[i][j] == 45:
                if G[i][j] >= G[i+1][j-1] and G[i][j] >= G[i-1][j+1]: #45
                    out[i][j] = G[i][j]
            if theta[i][j] == 90:
                if G[i][j] >= G[i-1][j] and G[i][j] >= G[i+1][j]: #90
                    out[i][j] = G[i][j]  
            if theta[i][j] == 135:
                if G[i][j] >= G[i-1][j-1] and G[i][j] >= G[i+1][j+1]: #180
                    out[i][j] = G[i][j]
                        

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] > high:
                strong_edges[i][j] = 1
            if low < img[i][j] <= high:
                weak_edges[i][j] = 1
    

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=np.bool)
            
    # Make new instances of arguments to leave the original
    # references intact
    weak_edge = np.copy(weak_edges)
    edges = np.copy(strong_edges)
    
    visited = {}
    for x in range(H):
        for y in range(W):
            visited[(x, y)] = 0
        
    for i in indices:
        queue = [] 
        queue.append(tuple(i)) 
        visited[tuple(i)] = 1
        
        while len(queue) != 0:
            s = queue.pop(0) 
            for j in  get_neighbors(s[0], s[1], H, W):
                if visited[j] == 0:
                    if weak_edge[j[0]][j[1]] == 1:
                        edges[j[0]][j[1]] = 1
                        queue.append(j)
                        visited[j] = 1
        
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(img)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    
    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W).
        
    Returns:
        accumulator: numpy array of shape (m, n).
        rhos: numpy array of shape (m, ).
        thetas: numpy array of shape (n, ).
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)
    
    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        
        for t in range(num_thetas):
            rho = round(x * cos_t[t] + y * sin_t[t]) + diag_len
            accumulator[int(rho), t] += 1



    return accumulator, rhos, thetas
