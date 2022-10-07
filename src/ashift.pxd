cdef extern from "ashift.h":

    cdef struct rect:
        double x1,y1,x2,y2;
        double width;
        double precision;
        
    float * shift(float width, float height, int input_line_count, rect rects[], int options);