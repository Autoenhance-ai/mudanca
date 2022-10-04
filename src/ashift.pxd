cdef extern from "ashift.h":

    cdef struct rect:
        double x1,y1,x2,y2;
        double width;
        double x,y;
        double theta;
        double dx,dy;
        double prec;
        double p;
        
    float * shift(float width, float height, int input_line_count, rect rects[]);