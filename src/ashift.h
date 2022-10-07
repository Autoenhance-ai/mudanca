typedef struct rect
{
  double x1,y1,x2,y2;  /* first and second point of the line segment */
  double width;        /* rectangle width */
  double precision;    /* precision of the line */
} rect;

extern float * shift(
    float width, float height,
    int input_line_count,
    rect rects[],
    int options
);