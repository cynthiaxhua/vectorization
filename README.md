# vectorization
various methods for image vectorization using Guo-Hall Thinning and Ramer-Douglas-Peuker
converts PNG images to lines defined by consecutive x,y coordinates

# convert_to_3_stroke:
Note: these processes assume input image is black line on white background
Takes a PNG image and outputs its vectorized format:
1) dilate and erode the image to group line segments together
2) convert to bitmap
3) trace bitmap to SVG
4) convert SVG to 3-stroke format
        
# get_opt_path:
Takes an unordered list of x,y coordinates and outputs the best ordering for the points so that they follow each other linearly:
1) create a cyclic neighbor-graph
2) iterate over starting points to see where we can minimize distance
        
# get_window_3_stroke
A windowed function for the 3-stroke conversion.
Takes a PNG image and window dimensions and outputs vectorized format:
1) use Guo-Hall thinning to reduce to a skeleton
2) get all non-zero points in skeleton
3) use DBSCAN clustering to find clusters of neighbors
4) interpret each cluster as a "stroke"
5) use get_opt_path on each cluster to find best-fit line
6) simplify best-fitting lines with Ramer-Douglas-Peuker algorithm
7) convert lines to strokes

# example
![example](https://github.com/cynthiaxhua/vectorization/demo.png "Vectorization")
