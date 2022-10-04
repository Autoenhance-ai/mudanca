/*
  TODO:
  - Adjust this header files

  This file is part of darktable,
  Copyright (C) 2016-2021 darktable developers.

  darktable is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  darktable is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "ashift.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#include "math.h"

#define DT_ALIGNED_ARRAY __attribute__((aligned(64)))

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

#ifndef MIN
#define MIN(a, b) (((a)<(b))?(a):(b))
#endif /* MIN */
#ifndef MAX
#define MAX(a, b) (((a)>(b))?(a):(b))
#endif  /* MAX */

/*----------------------------------------------------------------------------*/
/** Fatal error, print a message to standard-error output and exit.
 */
static void error(char * msg)
{
  fprintf(stderr,"LSD Error: %s\n",msg);
  exit(EXIT_FAILURE);
}

// Inspiration to this module comes from the program ShiftN (http://www.shiftn.de) by
// Marcus Hebel.

// Thanks to Marcus for his support when implementing part of the ShiftN functionality
// to darktable.

#define LSD_GAMMA 0.45                      // gamma correction to apply on raw images prior to line detection
#define ROTATION_RANGE 10                   // allowed min/max default range for rotation parameter
#define ROTATION_RANGE_SOFT 180             // allowed min/max range for rotation parameter with manual adjustment
#define LENSSHIFT_RANGE 1.0                 // allowed min/max default range for lensshift parameters
#define LENSSHIFT_RANGE_SOFT 2.0            // allowed min/max range for lensshift parameters with manual adjustment
#define SHEAR_RANGE 0.2                     // allowed min/max range for shear parameter
#define SHEAR_RANGE_SOFT 0.5                // allowed min/max range for shear parameter with manual adjustment
#define MIN_LINE_LENGTH 5                   // the minimum length of a line in pixels to be regarded as relevant
#define MAX_TANGENTIAL_DEVIATION 30         // by how many degrees a line may deviate from the +/-180 and +/-90 to be regarded as relevant
#define RANSAC_RUNS 400                     // how many iterations to run in ransac
#define RANSAC_EPSILON 2                    // starting value for ransac epsilon (in -log10 units)
#define RANSAC_EPSILON_STEP 1               // step size of epsilon optimization (log10 units)
#define RANSAC_ELIMINATION_RATIO 60         // percentage of lines we try to eliminate as outliers
#define RANSAC_OPTIMIZATION_STEPS 5         // home many steps to optimize epsilon
#define RANSAC_OPTIMIZATION_DRY_RUNS 50     // how man runs per optimization steps
#define RANSAC_HURDLE 5                     // hurdle rate: the number of lines below which we do a complete permutation instead of random sampling
#define MINIMUM_FITLINES 2                  // minimum number of lines needed for automatic parameter fit
#define NMS_EPSILON 1e-3                    // break criterion for Nelder-Mead simplex
#define NMS_SCALE 1.0                       // scaling factor for Nelder-Mead simplex
#define NMS_ITERATIONS 400                  // number of iterations for Nelder-Mead simplex
#define NMS_CROP_EPSILON 100.0              // break criterion for Nelder-Mead simplex on crop fitting
#define NMS_CROP_SCALE 0.5                  // scaling factor for Nelder-Mead simplex on crop fitting
#define NMS_CROP_ITERATIONS 100             // number of iterations for Nelder-Mead simplex on crop fitting
#define NMS_ALPHA 1.0                       // reflection coefficient for Nelder-Mead simplex
#define NMS_BETA 0.5                        // contraction coefficient for Nelder-Mead simplex
#define NMS_GAMMA 2.0                       // expansion coefficient for Nelder-Mead simplex
#define DEFAULT_F_LENGTH 28.0               // focal length we assume if no exif data are available

// define to get debugging output
// #undef ASHIFT_DEBUG

#define SQR(a) ((a) * (a))

#define CLAMP(x,min,max) \
    ({ \
        (x > max ? max : (x < min ? min : x)); \
    })

// For parameter optimization we are using the Nelder-Mead simplex method
// implemented by Michael F. Hutt.
#include "ashift_nmsimplex.c"

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */

typedef enum dt_iop_ashift_homodir_t
{
  ASHIFT_HOMOGRAPH_FORWARD,
  ASHIFT_HOMOGRAPH_INVERTED
} dt_iop_ashift_homodir_t;

typedef enum dt_iop_ashift_linetype_t
{
  ASHIFT_LINE_IRRELEVANT   = 0,       // the line is found to be not interesting
                                      // eg. too short, or not horizontal or vertical
  ASHIFT_LINE_RELEVANT     = 1 << 0,  // the line is relevant for us
  ASHIFT_LINE_DIRVERT      = 1 << 1,  // the line is (mostly) vertical, else (mostly) horizontal
  ASHIFT_LINE_SELECTED     = 1 << 2,  // the line is selected for fitting
  ASHIFT_LINE_VERTICAL_NOT_SELECTED   = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_DIRVERT,
  ASHIFT_LINE_HORIZONTAL_NOT_SELECTED = ASHIFT_LINE_RELEVANT,
  ASHIFT_LINE_VERTICAL_SELECTED = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_DIRVERT | ASHIFT_LINE_SELECTED,
  ASHIFT_LINE_HORIZONTAL_SELECTED = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_SELECTED,
  ASHIFT_LINE_MASK = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_DIRVERT | ASHIFT_LINE_SELECTED
} dt_iop_ashift_linetype_t;

typedef enum dt_iop_ashift_fitaxis_t
{
  ASHIFT_FIT_NONE          = 0,       // none
  ASHIFT_FIT_ROTATION      = 1 << 0,  // flag indicates to fit rotation angle
  ASHIFT_FIT_LENS_VERT     = 1 << 1,  // flag indicates to fit vertical lens shift
  ASHIFT_FIT_LENS_HOR      = 1 << 2,  // flag indicates to fit horizontal lens shift
  ASHIFT_FIT_SHEAR         = 1 << 3,  // flag indicates to fit shear parameter
  ASHIFT_FIT_LINES_VERT    = 1 << 4,  // use vertical lines for fitting
  ASHIFT_FIT_LINES_HOR     = 1 << 5,  // use horizontal lines for fitting
  ASHIFT_FIT_LENS_BOTH = ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LENS_HOR,
  ASHIFT_FIT_LINES_BOTH = ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_VERTICALLY = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LINES_VERT,
  ASHIFT_FIT_HORIZONTALLY = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LENS_HOR | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_BOTH = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LENS_HOR |
                    ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_VERTICALLY_NO_ROTATION = ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LINES_VERT,
  ASHIFT_FIT_HORIZONTALLY_NO_ROTATION = ASHIFT_FIT_LENS_HOR | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_BOTH_NO_ROTATION = ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LENS_HOR |
                                ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_BOTH_SHEAR = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LENS_HOR |
                    ASHIFT_FIT_SHEAR | ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_ROTATION_VERTICAL_LINES = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LINES_VERT,
  ASHIFT_FIT_ROTATION_HORIZONTAL_LINES = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_ROTATION_BOTH_LINES = ASHIFT_FIT_ROTATION | ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR,
  ASHIFT_FIT_FLIP = ASHIFT_FIT_LENS_VERT | ASHIFT_FIT_LENS_HOR | ASHIFT_FIT_LINES_VERT | ASHIFT_FIT_LINES_HOR
} dt_iop_ashift_fitaxis_t;

typedef enum dt_iop_ashift_nmsresult_t
{
  NMS_SUCCESS = 0,
  NMS_NOT_ENOUGH_LINES = 1,
  NMS_DID_NOT_CONVERGE = 2,
  NMS_INSANE = 3
} dt_iop_ashift_nmsresult_t;

typedef enum dt_iop_ashift_mode_t
{
  ASHIFT_MODE_GENERIC = 0, // $DESCRIPTION: "generic"
  ASHIFT_MODE_SPECIFIC = 1 // $DESCRIPTION: "specific"
} dt_iop_ashift_mode_t;

typedef struct dt_iop_ashift_params_t
{
  float rotation;    // $MIN: -ROTATION_RANGE_SOFT $MAX: ROTATION_RANGE_SOFT $DEFAULT: 0.0
  float lensshift_v; // $MIN: -LENSSHIFT_RANGE_SOFT $MAX: LENSSHIFT_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "lens shift (vertical)"
  float lensshift_h; // $MIN: -LENSSHIFT_RANGE_SOFT $MAX: LENSSHIFT_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "lens shift (horizontal)"
  float shear;       // $MIN: -SHEAR_RANGE_SOFT $MAX: SHEAR_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "shear"
  float f_length;    // $MIN: 1.0 $MAX: 2000.0 $DEFAULT: DEFAULT_F_LENGTH $DESCRIPTION: "focal length"
  float crop_factor; // $MIN: 0.5 $MAX: 10.0 $DEFAULT: 1.0 $DESCRIPTION: "crop factor"
  float orthocorr;   // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 100.0 $DESCRIPTION: "lens dependence"
  float aspect;      // $MIN: 0.5 $MAX: 2.0 $DEFAULT: 1.0 $DESCRIPTION: "aspect adjust"
  dt_iop_ashift_mode_t mode;     // $DEFAULT: ASHIFT_MODE_GENERIC $DESCRIPTION: "lens model"
  float cl;          // $DEFAULT: 0.0
  float cr;          // $DEFAULT: 1.0
  float ct;          // $DEFAULT: 0.0
  float cb;          // $DEFAULT: 1.0
} dt_iop_ashift_params_t;

typedef struct dt_iop_ashift_line_t
{
  float p1[3];
  float p2[3];
  float length;
  float width;
  float weight;
  dt_iop_ashift_linetype_t type;
  // homogeneous coordinates:
  float L[3];
} dt_iop_ashift_line_t;

typedef struct dt_iop_ashift_fit_params_t
{
  int params_count;
  dt_iop_ashift_linetype_t linetype;
  dt_iop_ashift_linetype_t linemask;
  dt_iop_ashift_line_t *lines;
  int lines_count;
  int width;
  int height;
  float weight;
  float f_length_kb;
  float orthocorr;
  float aspect;
  float rotation;
  float lensshift_v;
  float lensshift_h;
  float shear;
  float rotation_range;
  float lensshift_v_range;
  float lensshift_h_range;
  float shear_range;
} dt_iop_ashift_fit_params_t;

typedef struct dt_iop_ashift_cropfit_params_t
{
  int width;
  int height;
  float x;
  float y;
  float alpha;
  float homograph[3][3];
  float edges[4][3];
} dt_iop_ashift_cropfit_params_t;

typedef struct dt_iop_ashift_gui_data_t
{
  float rotation_range;
  float lensshift_v_range;
  float lensshift_h_range;
  float shear_range;
  dt_iop_ashift_line_t *lines;
  int lines_in_width;
  int lines_in_height;
  int lines_x_off;
  int lines_y_off;
  int lines_count;
  int vertical_count;
  int horizontal_count;
  int lines_version;
  float vertical_weight;
  float horizontal_weight;
  int buf_width;
  int buf_height;
  int buf_x_off;
  int buf_y_off;
  float buf_scale;
  int jobparams;
  int adjust_crop;
  float cl;	// shadow copy of dt_iop_ashift_data_t.cl
  float cr;	// shadow copy of dt_iop_ashift_data_t.cr
  float ct;	// shadow copy of dt_iop_ashift_data_t.ct
  float cb;	// shadow copy of dt_iop_ashift_data_t.cb
} dt_iop_ashift_gui_data_t;

#define generate_mat3inv_body(c_type, A, B)                                                                  \
  int mat3inv_##c_type(c_type *const dst, const c_type *const src)                                           \
  {                                                                                                          \
                                                                                                             \
    const c_type det = A(1, 1) * (A(3, 3) * A(2, 2) - A(3, 2) * A(2, 3))                                     \
                       - A(2, 1) * (A(3, 3) * A(1, 2) - A(3, 2) * A(1, 3))                                   \
                       + A(3, 1) * (A(2, 3) * A(1, 2) - A(2, 2) * A(1, 3));                                  \
                                                                                                             \
    const c_type epsilon = 1e-7f;                                                                            \
    if(fabs(det) < epsilon) return 1;                                                                        \
                                                                                                             \
    const c_type invDet = 1.0 / det;                                                                         \
                                                                                                             \
    B(1, 1) = invDet * (A(3, 3) * A(2, 2) - A(3, 2) * A(2, 3));                                              \
    B(1, 2) = -invDet * (A(3, 3) * A(1, 2) - A(3, 2) * A(1, 3));                                             \
    B(1, 3) = invDet * (A(2, 3) * A(1, 2) - A(2, 2) * A(1, 3));                                              \
                                                                                                             \
    B(2, 1) = -invDet * (A(3, 3) * A(2, 1) - A(3, 1) * A(2, 3));                                             \
    B(2, 2) = invDet * (A(3, 3) * A(1, 1) - A(3, 1) * A(1, 3));                                              \
    B(2, 3) = -invDet * (A(2, 3) * A(1, 1) - A(2, 1) * A(1, 3));                                             \
                                                                                                             \
    B(3, 1) = invDet * (A(3, 2) * A(2, 1) - A(3, 1) * A(2, 2));                                              \
    B(3, 2) = -invDet * (A(3, 2) * A(1, 1) - A(3, 1) * A(1, 2));                                             \
    B(3, 3) = invDet * (A(2, 2) * A(1, 1) - A(2, 1) * A(1, 2));                                              \
    return 0;                                                                                                \
  }

#define A(y, x) src[(y - 1) * 3 + (x - 1)]
#define B(y, x) dst[(y - 1) * 3 + (x - 1)]
/** inverts the given 3x3 matrix */
generate_mat3inv_body(float, A, B)

    int mat3inv(float *const dst, const float *const src)
{
  return mat3inv_float(dst, src);
}

generate_mat3inv_body(double, A, B)
#undef B
#undef A
#undef generate_mat3inv_body

// normalized product of two 3x1 vectors
// dst needs to be different from v1 and v2
static inline void vec3prodn(float *dst, const float *const v1, const float *const v2)
{
  const float l1 = v1[1] * v2[2] - v1[2] * v2[1];
  const float l2 = v1[2] * v2[0] - v1[0] * v2[2];
  const float l3 = v1[0] * v2[1] - v1[1] * v2[0];

  // normalize so that l1^2 + l2^2 + l3^3 = 1
  const float sq = sqrtf(l1 * l1 + l2 * l2 + l3 * l3);

  const float f = sq > 0.0f ? 1.0f / sq : 1.0f;

  dst[0] = l1 * f;
  dst[1] = l2 * f;
  dst[2] = l3 * f;
}

// normalize a 3x1 vector so that x^2 + y^2 + z^2 = 1
// dst and v may be the same
static inline void vec3norm(float *dst, const float *const v)
{
  const float sq = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

  // special handling for an all-zero vector
  const float f = sq > 0.0f ? 1.0f / sq : 1.0f;

  dst[0] = v[0] * f;
  dst[1] = v[1] * f;
  dst[2] = v[2] * f;
}

// normalize a 3x1 vector so that x^2 + y^2 = 1; a useful normalization for
// lines in homogeneous coordinates
// dst and v may be the same
static inline void vec3lnorm(float *dst, const float *const v)
{
  const float sq = sqrtf(v[0] * v[0] + v[1] * v[1]);

  // special handling for a point vector of the image center
  const float f = sq > 0.0f ? 1.0f / sq : 1.0f;

  dst[0] = v[0] * f;
  dst[1] = v[1] * f;
  dst[2] = v[2] * f;
}

// scalar product of two 3x1 vectors
static inline float vec3scalar(const float *const v1, const float *const v2)
{
  return (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]);
}

// check if 3x1 vector is (very close to) null
static inline int vec3isnull(const float *const v)
{
  const float eps = 1e-10f;
  return (fabsf(v[0]) < eps && fabsf(v[1]) < eps && fabsf(v[2]) < eps);
}

// process detectedlines according
// to this module's conventions
static int line_prcoess(
    const int lines_count,
    rect rects[],
    const int width, const int height,
    const int x_off, const int y_off,
    const float scale,
    dt_iop_ashift_line_t **alines,
    int *lcount, int *vcount, int *hcount,
    float *vweight, float *hweight
) {

  dt_iop_ashift_line_t *ashift_lines = NULL;

  int vertical_count = 0;
  int horizontal_count = 0;
  float vertical_weight = 0.0f;
  float horizontal_weight = 0.0f;

  // we count the lines that we really want to use
  int lct = 0;
  if(lines_count > 0)
  {
    // aggregate lines data into our own structures
    ashift_lines = (dt_iop_ashift_line_t *)malloc(sizeof(dt_iop_ashift_line_t) * lines_count);
    if(ashift_lines == NULL) goto error;

    for(int n = 0; n < lines_count; n++)
    {
      rect line = rects[n];

      const float x1 = line.x1;
      const float y1 = line.y1;
      const float x2 = line.x2;
      const float y2 = line.y2;

      // check for lines running along image borders and skip them.
      // these would likely be false-positives which could result
      // from any kind of processing artifacts
      if((fabsf(x1 - x2) < 1 && fmaxf(x1, x2) < 2)
         || (fabsf(x1 - x2) < 1 && fminf(x1, x2) > width - 3)
         || (fabsf(y1 - y2) < 1 && fmaxf(y1, y2) < 2)
         || (fabsf(y1 - y2) < 1 && fminf(y1, y2) > height - 3))
        continue;

      // line position in absolute coordinates
      float px1 = x_off + x1;
      float py1 = y_off + y1;
      float px2 = x_off + x2;
      float py2 = y_off + y2;

      // scale back to input buffer
      px1 /= scale;
      py1 /= scale;
      px2 /= scale;
      py2 /= scale;

      // store as homogeneous coordinates
      ashift_lines[lct].p1[0] = px1;
      ashift_lines[lct].p1[1] = py1;
      ashift_lines[lct].p1[2] = 1.0f;
      ashift_lines[lct].p2[0] = px2;
      ashift_lines[lct].p2[1] = py2;
      ashift_lines[lct].p2[2] = 1.0f;

      // calculate homogeneous coordinates of connecting line (defined by the two points)
      vec3prodn(ashift_lines[lct].L, ashift_lines[lct].p1, ashift_lines[lct].p2);

      // normalaze line coordinates so that x^2 + y^2 = 1
      // (this will always succeed as L is a real line connecting two real points)
      vec3lnorm(ashift_lines[lct].L, ashift_lines[lct].L);

      // length and width of rectangle (see LSD)
      ashift_lines[lct].length = sqrt((px2 - px1) * (px2 - px1) + (py2 - py1) * (py2 - py1));
      ashift_lines[lct].width = line.width / scale;

      // ...  and weight (= length * width * angle precision)
      const float weight = ashift_lines[lct].length * ashift_lines[lct].width * line.x;
      ashift_lines[lct].weight = weight;


      const float angle = atan2f(py2 - py1, px2 - px1) / M_PI * 180.0f;
      const int vertical = fabsf(fabsf(angle) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;
      const int horizontal = fabsf(fabsf(fabsf(angle) - 90.0f) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;

      const int relevant = ashift_lines[lct].length > MIN_LINE_LENGTH ? 1 : 0;

      // register type of line
      dt_iop_ashift_linetype_t type = ASHIFT_LINE_IRRELEVANT;
      if(vertical && relevant)
      {
        type = ASHIFT_LINE_VERTICAL_SELECTED;
        vertical_count++;
        vertical_weight += weight;
      }
      else if(horizontal && relevant)
      {
        type = ASHIFT_LINE_HORIZONTAL_SELECTED;
        horizontal_count++;
        horizontal_weight += weight;
      }
      ashift_lines[lct].type = type;

      // the next valid line
      lct++;
    }
  }
#ifdef ASHIFT_DEBUG
    printf("%d lines (vertical %d, horizontal %d, not relevant %d)\n", lines_count, vertical_count,
           horizontal_count, lct - vertical_count - horizontal_count);
    float xmin = FLT_MAX, xmax = FLT_MIN, ymin = FLT_MAX, ymax = FLT_MIN;
    for(int n = 0; n < lct; n++)
    {
      xmin = fmin(xmin, fmin(ashift_lines[n].p1[0], ashift_lines[n].p2[0]));
      xmax = fmax(xmax, fmax(ashift_lines[n].p1[0], ashift_lines[n].p2[0]));
      ymin = fmin(ymin, fmin(ashift_lines[n].p1[1], ashift_lines[n].p2[1]));
      ymax = fmax(ymax, fmax(ashift_lines[n].p1[1], ashift_lines[n].p2[1]));
      printf("x1 %.0f, y1 %.0f, x2 %.0f, y2 %.0f, length %.0f, width %f, X %f, Y %f, Z %f, type %d, scalars %f %f\n",
             ashift_lines[n].p1[0], ashift_lines[n].p1[1], ashift_lines[n].p2[0], ashift_lines[n].p2[1],
             ashift_lines[n].length, ashift_lines[n].width,
             ashift_lines[n].L[0], ashift_lines[n].L[1], ashift_lines[n].L[2], ashift_lines[n].type,
             vec3scalar(ashift_lines[n].p1, ashift_lines[n].L),
             vec3scalar(ashift_lines[n].p2, ashift_lines[n].L));
    }
    printf("xmin %.0f, xmax %.0f, ymin %.0f, ymax %.0f\n", xmin, xmax, ymin, ymax);
#endif

  // store results in provided locations
  *lcount = lct;
  *vcount = vertical_count;
  *vweight = vertical_weight;
  *hcount = horizontal_count;
  *hweight = horizontal_weight;
  *alines = ashift_lines;

  // free intermediate buffers
  return lct > 0 ? TRUE : FALSE;

error:
  return FALSE;
}

// swap two integer values
static inline void swap(int *a, int *b)
{
  int tmp = *a;
  *a = *b;
  *b = tmp;
}

// do complete permutations
static int quickperm(int *a, int *p, const int N, int *i)
{
  if(*i >= N) return FALSE;

  p[*i]--;
  int j = (*i % 2 == 1) ? p[*i] : 0;
  swap(&a[j], &a[*i]);
  *i = 1;
  while(p[*i] == 0)
  {
    p[*i] = *i;
    (*i)++;
  }
  return TRUE;
}

// Fisher-Yates shuffle
static void shuffle(int *a, const int N)
{
  for(int i = 0; i < N; i++)
  {
    int j = i + rand() % (N - i);
    swap(&a[j], &a[i]);
  }
}

// factorial function
static int fact(const int n)
{
  return (n == 1 ? 1 : n * fact(n - 1));
}

// We use a pseudo-RANSAC algorithm to elminiate ouliers from our set of lines. The
// original RANSAC works on linear optimization problems. Our model is nonlinear. We
// take advantage of the fact that lines interesting for our model are vantage lines
// that meet in one vantage point for each subset of lines (vertical/horizontal).
// Strategy: we construct a model by (random) sampling within the subset of lines and
// calculate the vantage point. Then we check the "distance" of all other lines to the
// vantage point. The model that gives highest number of lines combined with the highest
// total weight and lowest overall "distance" wins.
// Disadvantage: compared to the original RANSAC we don't get any model parameters that
// we could use for the following NMS fit.
// Self-tuning: we optimize "epsilon", the hurdle rate to reject a line as an outlier,
// by a number of dry runs first. The target average percentage value of lines to eliminate as
// outliers (without judging on the quality of the model) is given by RANSAC_ELIMINATION_RATIO,
// note: the actual percentage of outliers removed in the final run will be lower because we
// will finally look for the best quality model with the optimized epsilon and that quality value also
// encloses the number of good lines
static void ransac(const dt_iop_ashift_line_t *lines, int *index_set, int *inout_set,
                  const int set_count, const float total_weight, const int xmin, const int xmax,
                  const int ymin, const int ymax)
{
  if(set_count < 3) return;

  const size_t set_size = set_count * sizeof(int);
  int *best_set = malloc(set_size);
  memcpy(best_set, index_set, set_size);
  int *best_inout = calloc(1, set_size);

  float best_quality = 0.0f;

  // hurdle value epsilon for rejecting a line as an outlier will be self-tuning
  // in a number of dry runs
  float epsilon = powf(10.0f, -RANSAC_EPSILON);
  float epsilon_step = RANSAC_EPSILON_STEP;
  // some accounting variables for self-tuning
  int lines_eliminated = 0;
  int valid_runs = 0;

  // number of runs to optimize epsilon
  const int optiruns = RANSAC_OPTIMIZATION_STEPS * RANSAC_OPTIMIZATION_DRY_RUNS;
  // go for complete permutations on small set sizes, else for random sample consensus
  const int riter = (set_count > RANSAC_HURDLE) ? RANSAC_RUNS : fact(set_count);

  // some data needed for quickperm
  int *perm = malloc(sizeof(int) * (set_count + 1));
  for(int n = 0; n < set_count + 1; n++) perm[n] = n;
  int piter = 1;

  // inout holds good/bad qualification for each line
  int *inout = malloc(set_size);

  for(int r = 0; r < optiruns + riter; r++)
  {
    // get random or systematic variation of index set
    if(set_count > RANSAC_HURDLE || r < optiruns)
      shuffle(index_set, set_count);
    else
      (void)quickperm(index_set, perm, set_count, &piter);

    // summed quality evaluation of this run
    float quality = 0.0f;

    // we build a model ouf of the first two lines
    const float *L1 = lines[index_set[0]].L;
    const float *L2 = lines[index_set[1]].L;

    // get intersection point (ideally a vantage point)
    float V[3];
    vec3prodn(V, L1, L2);

    // catch special cases:
    // a) L1 and L2 are identical -> V is NULL -> no valid vantage point
    // b) vantage point lies inside image frame (no chance to correct for this case)
    if(vec3isnull(V) ||
       (fabsf(V[2]) > 0.0f &&
        V[0]/V[2] >= xmin &&
        V[1]/V[2] >= ymin &&
        V[0]/V[2] <= xmax &&
        V[1]/V[2] <= ymax))
    {
      // no valid model
      quality = 0.0f;
    }
    else
    {
      // valid model

      // normalize V so that x^2 + y^2 + z^2 = 1
      vec3norm(V, V);

      // the two lines constituting the model are part of the set
      inout[0] = 1;
      inout[1] = 1;

      // go through all remaining lines, check if they are within the model, and
      // mark that fact in inout[].
      // summarize a quality parameter for all lines within the model
      for(int n = 2; n < set_count; n++)
      {
        // L is normalized so that x^2 + y^2 = 1
        const float *L3 = lines[index_set[n]].L;

        // we take the absolute value of the dot product of V and L as a measure
        // of the "distance" between point and line. Note that this is not the real euclidean
        // distance but - with the given normalization - just a pragmatically selected number
        // that goes to zero if V lies on L and increases the more V and L are apart
        const float d = fabsf(vec3scalar(V, L3));

        // depending on d we either include or exclude the point from the set
        inout[n] = (d < epsilon) ? 1 : 0;

        float q;

        if(inout[n] == 1)
        {
          // a quality parameter that depends 1/3 on the number of lines within the model,
          // 1/3 on their weight, and 1/3 on their weighted distance d to the vantage point
          q = 0.33f / (float)set_count
              + 0.33f * lines[index_set[n]].weight / total_weight
              + 0.33f * (1.0f - d / epsilon) * (float)set_count * lines[index_set[n]].weight / total_weight;
        }
        else
        {
          q = 0.0f;
          lines_eliminated++;
        }

        quality += q;
      }
      valid_runs++;
    }

    if(r < optiruns)
    {
      // on last run of each self-tuning step
      if((r % RANSAC_OPTIMIZATION_DRY_RUNS) == (RANSAC_OPTIMIZATION_DRY_RUNS - 1) && (valid_runs > 0))
      {
#ifdef ASHIFT_DEBUG
        printf("ransac self-tuning (run %d): epsilon %f", r, epsilon);
#endif
        // average ratio of lines that we eliminated with the given epsilon
        float ratio = 100.0f * (float)lines_eliminated / ((float)set_count * valid_runs);
        // adjust epsilon accordingly
        if(ratio < RANSAC_ELIMINATION_RATIO)
          epsilon = powf(10.0f, log10(epsilon) - epsilon_step);
        else if(ratio > RANSAC_ELIMINATION_RATIO)
          epsilon = powf(10.0f, log10(epsilon) + epsilon_step);
#ifdef ASHIFT_DEBUG
        printf(" (elimination ratio %f) -> %f\n", ratio, epsilon);
#endif
        // reduce step-size for next optimization round
        epsilon_step /= 2.0f;
        lines_eliminated = 0;
        valid_runs = 0;
      }
    }
    else
    {
      // in the "real" runs check against the best model found so far
      if(quality > best_quality)
      {
        memcpy(best_set, index_set, set_size);
        memcpy(best_inout, inout, set_size);
        best_quality = quality;
      }
    }

#ifdef ASHIFT_DEBUG
    // report some statistics
    int count = 0, lastcount = 0;
    for(int n = 0; n < set_count; n++) count += best_inout[n];
    for(int n = 0; n < set_count; n++) lastcount += inout[n];
    printf("ransac run %d: best qual %.6f, eps %.6f, line count %d of %d (this run: qual %.5f, count %d (%2f%%))\n", r,
           best_quality, epsilon, count, set_count, quality, lastcount, 100.0f * lastcount / (float)set_count);
#endif
  }

  // store back best set
  memcpy(index_set, best_set, set_size);
  memcpy(inout_set, best_inout, set_size);

  free(inout);
  free(perm);
  free(best_inout);
  free(best_set);
}

// try to clean up structural data by eliminating outliers and thereby increasing
// the chance of a convergent fitting
static int _remove_outliers(dt_iop_ashift_gui_data_t *g)
{
  const int width = g->lines_in_width;
  const int height = g->lines_in_height;
  const int xmin = g->lines_x_off;
  const int ymin = g->lines_y_off;
  const int xmax = xmin + width;
  const int ymax = ymin + height;

  // holds the index set of lines we want to work on
  int *lines_set = malloc(sizeof(int) * g->lines_count);
  // holds the result of ransac
  int *inout_set = malloc(sizeof(int) * g->lines_count);

  // some accounting variables
  int vnb = 0, vcount = 0;
  int hnb = 0, hcount = 0;

  // just to be on the safe side
  if(g->lines == NULL) goto error;

  // generate index list for the vertical lines
  for(int n = 0; n < g->lines_count; n++)
  {
    // is this a selected vertical line?
    if((g->lines[n].type & ASHIFT_LINE_MASK) != ASHIFT_LINE_VERTICAL_SELECTED)
      continue;

    lines_set[vnb] = n;
    inout_set[vnb] = 0;
    vnb++;
  }

  // it only makes sense to call ransac if we have more than two lines
  if(vnb > 2)
    ransac(g->lines, lines_set, inout_set, vnb, g->vertical_weight,
           xmin, xmax, ymin, ymax);

  // adjust line selected flag according to the ransac results
  for(int n = 0; n < vnb; n++)
  {
    const int m = lines_set[n];
    if(inout_set[n] == 1)
    {
      g->lines[m].type |= ASHIFT_LINE_SELECTED;
      vcount++;
    }
    else
      g->lines[m].type &= ~ASHIFT_LINE_SELECTED;
  }
  // update number of vertical lines
  g->vertical_count = vcount;
  g->lines_version++;

  // now generate index list for the horizontal lines
  for(int n = 0; n < g->lines_count; n++)
  {
    // is this a selected horizontal line?
    if((g->lines[n].type & ASHIFT_LINE_MASK) != ASHIFT_LINE_HORIZONTAL_SELECTED)
      continue;

    lines_set[hnb] = n;
    inout_set[hnb] = 0;
    hnb++;
  }

  // it only makes sense to call ransac if we have more than two lines
  if(hnb > 2)
    ransac(g->lines, lines_set, inout_set, hnb, g->horizontal_weight,
           xmin, xmax, ymin, ymax);

  // adjust line selected flag according to the ransac results
  for(int n = 0; n < hnb; n++)
  {
    const int m = lines_set[n];
    if(inout_set[n] == 1)
    {
      g->lines[m].type |= ASHIFT_LINE_SELECTED;
      hcount++;
    }
    else
      g->lines[m].type &= ~ASHIFT_LINE_SELECTED;
  }
  // update number of horizontal lines
  g->horizontal_count = hcount;
  g->lines_version++;

  free(inout_set);
  free(lines_set);

  return TRUE;

error:
  free(inout_set);
  free(lines_set);
  return FALSE;
}

// utility function to map a variable in [min; max] to [-INF; + INF]
static inline double logit(double x, double min, double max)
{
  const double eps = 1.0e-6;
  // make sure p does not touch the borders of its definition area,
  // not critical for data accuracy as logit() is only used on initial fit parameters
  double p = CLAMP((x - min) / (max - min), eps, 1.0 - eps);

  return (2.0 * atanh(2.0 * p - 1.0));
}

// inverted function to logit()
static inline double ilogit(double L, double min, double max)
{
  double p = 0.5 * (1.0 + tanh(0.5 * L));

  return (p * (max - min) + min);
}

#define MAT3SWAP(a, b) { float (*tmp)[3] = (a); (a) = (b); (b) = tmp; }

static void homography(float *homograph, const float angle, const float shift_v, const float shift_h,
                       const float shear, const float f_length_kb, const float orthocorr, const float aspect,
                       const int width, const int height, dt_iop_ashift_homodir_t dir)
{
  // calculate homograph that combines all translations, rotations
  // and warping into one single matrix operation.
  // this is heavily leaning on ShiftN where the homographic matrix expects
  // input in (y : x : 1) format. in the darktable world we want to keep the
  // (x : y : 1) convention. therefore we need to flip coordinates first and
  // make sure that output is in correct format after corrections are applied.

  const float u = width;
  const float v = height;

  const float phi = M_PI * angle / 180.0f;
  const float cosi = cosf(phi);
  const float sini = sinf(phi);
  const float ascale = sqrtf(aspect);

  // most of this comes from ShiftN
  const float f_global = f_length_kb;
  const float horifac = 1.0f - orthocorr / 100.0f;
  const float exppa_v = expf(shift_v);
  const float fdb_v = f_global / (14.4f + (v / u - 1) * 7.2f);
  const float rad_v = fdb_v * (exppa_v - 1.0f) / (exppa_v + 1.0f);
  const float alpha_v = CLAMP(atanf(rad_v), -1.5f, 1.5f);
  const float rt_v = sinf(0.5f * alpha_v);
  const float r_v = fmaxf(0.1f, 2.0f * (horifac - 1.0f) * rt_v * rt_v + 1.0f);

  const float vertifac = 1.0f - orthocorr / 100.0f;
  const float exppa_h = expf(shift_h);
  const float fdb_h = f_global / (14.4f + (u / v - 1) * 7.2f);
  const float rad_h = fdb_h * (exppa_h - 1.0f) / (exppa_h + 1.0f);
  const float alpha_h = CLAMP(atanf(rad_h), -1.5f, 1.5f);
  const float rt_h = sinf(0.5f * alpha_h);
  const float r_h = fmaxf(0.1f, 2.0f * (vertifac - 1.0f) * rt_h * rt_h + 1.0f);


  // three intermediate buffers for matrix calculation ...
  float m1[3][3], m2[3][3], m3[3][3];

  // ... and some pointers to handle them more intuitively
  float (*mwork)[3] = m1;
  float (*minput)[3] = m2;
  float (*moutput)[3] = m3;

  // Step 1: flip x and y coordinates (see above)
  memset(minput, 0, sizeof(float) * 9);
  minput[0][1] = 1.0f;
  minput[1][0] = 1.0f;
  minput[2][2] = 1.0f;


  // Step 2: rotation of image around its center
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = cosi;
  mwork[0][1] = -sini;
  mwork[1][0] = sini;
  mwork[1][1] = cosi;
  mwork[0][2] = -0.5f * v * cosi + 0.5f * u * sini + 0.5f * v;
  mwork[1][2] = -0.5f * v * sini - 0.5f * u * cosi + 0.5f * u;
  mwork[2][2] = 1.0f;

  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 3: apply shearing
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = 1.0f;
  mwork[0][1] = shear;
  mwork[1][1] = 1.0f;
  mwork[1][0] = shear;
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 4: apply vertical lens shift effect
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = exppa_v;
  mwork[1][0] = 0.5f * ((exppa_v - 1.0f) * u) / v;
  mwork[1][1] = 2.0f * exppa_v / (exppa_v + 1.0f);
  mwork[1][2] = -0.5f * ((exppa_v - 1.0f) * u) / (exppa_v + 1.0f);
  mwork[2][0] = (exppa_v - 1.0f) / v;
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 5: horizontal compression
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = 1.0f;
  mwork[1][1] = r_v;
  mwork[1][2] = 0.5f * u * (1.0f - r_v);
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 6: flip x and y back again
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][1] = 1.0f;
  mwork[1][0] = 1.0f;
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // from here output vectors would be in (x : y : 1) format

  // Step 7: now we can apply horizontal lens shift with the same matrix format as above
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = exppa_h;
  mwork[1][0] = 0.5f * ((exppa_h - 1.0f) * v) / u;
  mwork[1][1] = 2.0f * exppa_h / (exppa_h + 1.0f);
  mwork[1][2] = -0.5f * ((exppa_h - 1.0f) * v) / (exppa_h + 1.0f);
  mwork[2][0] = (exppa_h - 1.0f) / u;
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 8: vertical compression
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = 1.0f;
  mwork[1][1] = r_h;
  mwork[1][2] = 0.5f * v * (1.0f - r_h);
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // Step 9: apply aspect ratio scaling
  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = 1.0f * ascale;
  mwork[1][1] = 1.0f / ascale;
  mwork[2][2] = 1.0f;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);

  // Step 10: find x/y offsets and apply according correction so that
  // no negative coordinates occur in output vector
  float umin = FLT_MAX, vmin = FLT_MAX;
  // visit all four corners
  for(int y = 0; y < height; y += height - 1)
    for(int x = 0; x < width; x += width - 1)
    {
      float pi[3], po[3];
      pi[0] = x;
      pi[1] = y;
      pi[2] = 1.0f;
      // moutput expects input in (x:y:1) format and gives output as (x:y:1)
      mat3mulv(po, (float *)moutput, pi);
      umin = fmin(umin, po[0] / po[2]);
      vmin = fmin(vmin, po[1] / po[2]);
    }

  memset(mwork, 0, sizeof(float) * 9);
  mwork[0][0] = 1.0f;
  mwork[1][1] = 1.0f;
  mwork[2][2] = 1.0f;
  mwork[0][2] = -umin;
  mwork[1][2] = -vmin;

  // moutput (of last calculation) -> minput
  MAT3SWAP(minput, moutput);
  // multiply mwork * minput -> moutput
  mat3mul((float *)moutput, (float *)mwork, (float *)minput);


  // on request we either keep the final matrix for forward conversions
  // or produce an inverted matrix for backward conversions
  if(dir == ASHIFT_HOMOGRAPH_FORWARD)
  {
    // we have what we need -> copy it to the right place
    memcpy(homograph, moutput, sizeof(float) * 9);
  }
  else
  {
    // generate inverted homograph (mat3inv function defined in colorspaces.c)
    if(mat3inv((float *)homograph, (float *)moutput))
    {
      // in case of error we set to unity matrix
      memset(mwork, 0, sizeof(float) * 9);
      mwork[0][0] = 1.0f;
      mwork[1][1] = 1.0f;
      mwork[2][2] = 1.0f;
      memcpy(homograph, mwork, sizeof(float) * 9);
    }
  }
}
#undef MAT3SWAP


// helper function for simplex() return quality parameter for the given model
// strategy:
//    * generate homography matrix out of fixed parameters and fitting parameters
//    * apply homography to all end points of affected lines
//    * generate new line out of transformed end points
//    * calculate scalar product s of line with perpendicular axis
//    * sum over weighted s^2 values
static double model_fitness(double *params, void *data)
{
  dt_iop_ashift_fit_params_t *fit = (dt_iop_ashift_fit_params_t *)data;

  // just for convenience: get shorter names
  dt_iop_ashift_line_t *lines = fit->lines;
  const int lines_count = fit->lines_count;
  const int width = fit->width;
  const int height = fit->height;
  const float f_length_kb = fit->f_length_kb;
  const float orthocorr = fit->orthocorr;
  const float aspect = fit->aspect;

  float rotation = fit->rotation;
  float lensshift_v = fit->lensshift_v;
  float lensshift_h = fit->lensshift_h;
  float shear = fit->shear;
  float rotation_range = fit->rotation_range;
  float lensshift_v_range = fit->lensshift_v_range;
  float lensshift_h_range = fit->lensshift_h_range;
  float shear_range = fit->shear_range;

  int pcount = 0;

  // fill in fit parameters from params[]. Attention: order matters!!!
  if(isnan(rotation))
  {
    rotation = ilogit(params[pcount], -rotation_range, rotation_range);
    pcount++;
  }

  if(isnan(lensshift_v))
  {
    lensshift_v = ilogit(params[pcount], -lensshift_v_range, lensshift_v_range);
    pcount++;
  }

  if(isnan(lensshift_h))
  {
    lensshift_h = ilogit(params[pcount], -lensshift_h_range, lensshift_h_range);
    pcount++;
  }

  if(isnan(shear))
  {
    shear = ilogit(params[pcount], -shear_range, shear_range);
    pcount++;
  }

  assert(pcount == fit->params_count);

  // the possible reference axes
  const float Av[3] = { 1.0f, 0.0f, 0.0f };
  const float Ah[3] = { 0.0f, 1.0f, 0.0f };

  // generate homograph out of the parameters
  float homograph[3][3];
  homography((float *)homograph, rotation, lensshift_v, lensshift_h, shear, f_length_kb,
             orthocorr, aspect, width, height, ASHIFT_HOMOGRAPH_FORWARD);

  // accounting variables
  double sumsq_v = 0.0;
  double sumsq_h = 0.0;
  double weight_v = 0.0;
  double weight_h = 0.0;
  int count_v = 0;
  int count_h = 0;
  int count = 0;

  // iterate over all lines
  for(int n = 0; n < lines_count; n++)
  {
    // check if this is a line which we must skip
    if((lines[n].type & fit->linemask) != fit->linetype)
      continue;

    // the direction of this line (vertical?)
    const int isvertical = lines[n].type & ASHIFT_LINE_DIRVERT;

    // select the perpendicular reference axis
    const float *A = isvertical ? Ah : Av;

    // apply homographic transformation to the end points
    float P1[3], P2[3];
    mat3mulv(P1, (float *)homograph, lines[n].p1);
    mat3mulv(P2, (float *)homograph, lines[n].p2);

    // get line connecting the two points
    float L[3];
    vec3prodn(L, P1, P2);

    // normalize L so that x^2 + y^2 = 1; makes sure that
    // y^2 = 1 / (1 + m^2) and x^2 = m^2 / (1 + m^2) with m defining the slope of the line
    vec3lnorm(L, L);

    // get scalar product of line L with orthogonal axis A -> gives 0 if line is perpendicular
    float s = vec3scalar(L, A);

    // sum up weighted s^2 for both directions individually
    sumsq_v += isvertical ? s * s * lines[n].weight : 0.0;
    weight_v  += isvertical ? lines[n].weight : 0.0;
    count_v += isvertical ? 1 : 0;
    sumsq_h += !isvertical ? s * s * lines[n].weight : 0.0;
    weight_h  += !isvertical ? lines[n].weight : 0.0;
    count_h += !isvertical ? 1 : 0;
    count++;
  }

  const double v = weight_v > 0.0f && count > 0 ? sumsq_v / weight_v * (float)count_v / count : 0.0;
  const double h = weight_h > 0.0f && count > 0 ? sumsq_h / weight_h * (float)count_h / count : 0.0;

  double sum = sqrt(1.0 - (1.0 - v) * (1.0 - h)) * 1.0e6;
  //double sum = sqrt(v + h) * 1.0e6;

#ifdef ASHIFT_DEBUG
  printf("fitness with rotation %f, lensshift_v %f, lensshift_h %f, shear %f -> lines %d, quality %10f\n",
         rotation, lensshift_v, lensshift_h, shear, count, sum);
#endif

  return sum;
}

// setup all data structures for fitting and call NM simplex
static dt_iop_ashift_nmsresult_t nmsfit(dt_iop_ashift_gui_data_t *g, dt_iop_ashift_params_t *p, dt_iop_ashift_fitaxis_t dir)
{
  if(!g->lines) return NMS_NOT_ENOUGH_LINES;
  if(dir == ASHIFT_FIT_NONE) return NMS_SUCCESS;

  double params[4];
  int pcount = 0;
  int enough_lines = TRUE;

  // initialize fit parameters
  dt_iop_ashift_fit_params_t fit;
  fit.lines = g->lines;
  fit.lines_count = g->lines_count;
  fit.width = g->lines_in_width;
  fit.height = g->lines_in_height;
  fit.f_length_kb = (p->mode == ASHIFT_MODE_GENERIC) ? DEFAULT_F_LENGTH : p->f_length * p->crop_factor;
  fit.orthocorr = (p->mode == ASHIFT_MODE_GENERIC) ? 0.0f : p->orthocorr;
  fit.aspect = (p->mode == ASHIFT_MODE_GENERIC) ? 1.0f : p->aspect;
  fit.rotation = p->rotation;
  fit.lensshift_v = p->lensshift_v;
  fit.lensshift_h = p->lensshift_h;
  fit.shear = p->shear;
  fit.rotation_range = g->rotation_range;
  fit.lensshift_v_range = g->lensshift_v_range;
  fit.lensshift_h_range = g->lensshift_h_range;
  fit.shear_range = g->shear_range;
  fit.linetype = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_SELECTED;
  fit.linemask = ASHIFT_LINE_MASK;
  fit.params_count = 0;
  fit.weight = 0.0f;

  // if the image is flipped and if we do not want to fit both lens shift
  // directions or none at all, then we need to change direction
  dt_iop_ashift_fitaxis_t mdir = dir;

  // prepare fit structure and starting parameters for simplex fit.
  // note: the sequence of parameters in params[] needs to match the
  // respective order in dt_iop_ashift_fit_params_t. Parameters which are
  // to be fittet are marked with NAN in the fit structure. Non-NAN
  // parameters are assumed to be constant.
  if(mdir & ASHIFT_FIT_ROTATION)
  {
    // we fit rotation
    fit.params_count++;
    params[pcount] = logit(fit.rotation, -fit.rotation_range, fit.rotation_range);
    pcount++;
    fit.rotation = NAN;
  }

  if(mdir & ASHIFT_FIT_LENS_VERT)
  {
    // we fit vertical lens shift
    fit.params_count++;
    params[pcount] = logit(fit.lensshift_v, -fit.lensshift_v_range, fit.lensshift_v_range);
    pcount++;
    fit.lensshift_v = NAN;
  }

  if(mdir & ASHIFT_FIT_LENS_HOR)
  {
    // we fit horizontal lens shift
    fit.params_count++;
    params[pcount] = logit(fit.lensshift_h, -fit.lensshift_h_range, fit.lensshift_h_range);
    pcount++;
    fit.lensshift_h = NAN;
  }

  if(mdir & ASHIFT_FIT_SHEAR)
  {
    // we fit the shear parameter
    fit.params_count++;
    params[pcount] = logit(fit.shear, -fit.shear_range, fit.shear_range);
    pcount++;
    fit.shear = NAN;
  }

  if(mdir & ASHIFT_FIT_LINES_VERT)
  {
    // we use vertical lines for fitting
    fit.linetype |= ASHIFT_LINE_DIRVERT;
    fit.weight += g->vertical_weight;
    enough_lines = enough_lines && (g->vertical_count >= MINIMUM_FITLINES);
  }

  if(mdir & ASHIFT_FIT_LINES_HOR)
  {
    // we use horizontal lines for fitting
    fit.linetype |= 0;
    fit.weight += g->horizontal_weight;
    enough_lines = enough_lines && (g->horizontal_count >= MINIMUM_FITLINES);
  }

  // this needs to come after ASHIFT_FIT_LINES_VERT and ASHIFT_FIT_LINES_HOR
  if((mdir & ASHIFT_FIT_LINES_BOTH) == ASHIFT_FIT_LINES_BOTH)
  {
    // if we use fitting in both directions we need to
    // adjust fit.linetype and fit.linemask to match all selected lines
    fit.linetype = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_SELECTED;
    fit.linemask = ASHIFT_LINE_RELEVANT | ASHIFT_LINE_SELECTED;
  }

  // error case: we do not run simplex if there are not enough lines
  if(!enough_lines)
  {
#ifdef ASHIFT_DEBUG
    printf("optimization not possible: insufficient number of lines\n");
#endif
    return NMS_NOT_ENOUGH_LINES;
  }

  // start the simplex fit
  int iter = simplex(model_fitness, params, fit.params_count, NMS_EPSILON, NMS_SCALE, NMS_ITERATIONS, NULL, (void*)&fit);

  // error case: the fit did not converge
  if(iter >= NMS_ITERATIONS)
  {
#ifdef ASHIFT_DEBUG
    printf("optimization not successful: maximum number of iterations reached (%d)\n", iter);
#endif
    return NMS_DID_NOT_CONVERGE;
  }

  // fit was successful: now consolidate the results (order matters!!!)
  pcount = 0;
  fit.rotation = isnan(fit.rotation) ? ilogit(params[pcount++], -fit.rotation_range, fit.rotation_range) : fit.rotation;
  fit.lensshift_v = isnan(fit.lensshift_v) ? ilogit(params[pcount++], -fit.lensshift_v_range, fit.lensshift_v_range) : fit.lensshift_v;
  fit.lensshift_h = isnan(fit.lensshift_h) ? ilogit(params[pcount++], -fit.lensshift_h_range, fit.lensshift_h_range) : fit.lensshift_h;
  fit.shear = isnan(fit.shear) ? ilogit(params[pcount++], -fit.shear_range, fit.shear_range) : fit.shear;
#ifdef ASHIFT_DEBUG
  printf("params after optimization (%d iterations): rotation %f, lensshift_v %f, lensshift_h %f, shear %f\n",
         iter, fit.rotation, fit.lensshift_v, fit.lensshift_h, fit.shear);
#endif

  // sanity check: in case of extreme values the image gets distorted so strongly that it spans an insanely huge area. we check that
  // case and assume values that increase the image area by more than a factor of 4 as being insane.
  float homograph[3][3];
  homography((float *)homograph, fit.rotation, fit.lensshift_v, fit.lensshift_h, fit.shear, fit.f_length_kb,
             fit.orthocorr, fit.aspect, fit.width, fit.height, ASHIFT_HOMOGRAPH_FORWARD);

  // visit all four corners and find maximum span
  float xm = FLT_MAX, xM = -FLT_MAX, ym = FLT_MAX, yM = -FLT_MAX;
  for(int y = 0; y < fit.height; y += fit.height - 1)
    for(int x = 0; x < fit.width; x += fit.width - 1)
    {
      float pi[3], po[3];
      pi[0] = x;
      pi[1] = y;
      pi[2] = 1.0f;
      mat3mulv(po, (float *)homograph, pi);
      po[0] /= po[2];
      po[1] /= po[2];
      xm = fmin(xm, po[0]);
      ym = fmin(ym, po[1]);
      xM = fmax(xM, po[0]);
      yM = fmax(yM, po[1]);
    }

  if((xM - xm) * (yM - ym) > 4.0f * fit.width * fit.height)
  {
#ifdef ASHIFT_DEBUG
    printf("optimization not successful: degenerate case with area growth factor (%f) exceeding limits\n",
           (xM - xm) * (yM - ym) / (fit.width * fit.height));
#endif
    return NMS_INSANE;
  }

  // now write the results into structure p
  p->rotation = fit.rotation;
  p->lensshift_v = fit.lensshift_v;
  p->lensshift_h = fit.lensshift_h;
  p->shear = fit.shear;
  return NMS_SUCCESS;
}

static void _clear_shadow_crop_box(dt_iop_ashift_gui_data_t *g)
{
  // reset the crop to the full image
  g->cl = 0.0f;
  g->cr = 1.0f;
  g->ct = 0.0f;
  g->cb = 1.0f;
}

static inline void _commit_crop_box(dt_iop_ashift_params_t *p, dt_iop_ashift_gui_data_t *g)
{
  // copy shadow values for crop box into actual parameters
  p->cl = g->cl;
  p->cr = g->cr;
  p->ct = g->ct;
  p->cb = g->cb;
}

// function to keep crop fitting parameters within constraints
static void crop_constraint(double *params, int pcount)
{
  if(pcount > 0) params[0] = fabs(params[0]);
  if(pcount > 1) params[1] = fabs(params[1]);
  if(pcount > 2) params[2] = fabs(params[2]);

  if(pcount > 0 && params[0] > 1.0) params[0] = 1.0 - params[0];
  if(pcount > 1 && params[1] > 1.0) params[1] = 1.0 - params[1];
  if(pcount > 2 && params[2] > 0.5*M_PI) params[2] = 0.5*M_PI - params[2];
}

// helper function for getting the best fitting crop area;
// returns the negative area of the largest rectangle that fits within the
// defined image with a given rectangle's center and its aspect angle;
// the trick: the rectangle center coordinates are given in the input
// image coordinates so we know for sure that it also lies within the image after
// conversion to the output coordinates
static double crop_fitness(double *params, void *data)
{
  dt_iop_ashift_cropfit_params_t *cropfit = (dt_iop_ashift_cropfit_params_t *)data;

  const float wd = cropfit->width;
  const float ht = cropfit->height;

  // get variable and constant parameters, respectively
  const float x = isnan(cropfit->x) ? params[0] : cropfit->x;
  const float y = isnan(cropfit->y) ? params[1] : cropfit->y;
  const float alpha = isnan(cropfit->alpha) ? params[2] : cropfit->alpha;

  // the center of the rectangle in input image coordinates
  const float Pc[3] = { x * wd, y * ht, 1.0f };

  // convert to the output image coordinates and normalize
  float P[3];
  mat3mulv(P, (float *)cropfit->homograph, Pc);
  P[0] /= P[2];
  P[1] /= P[2];
  P[2] = 1.0f;

  // two auxiliary points (some arbitrary distance away from P) to construct the diagonals
  const float Pa[2][3] = { { P[0] + 10.0f * cosf(alpha), P[1] + 10.0f * sinf(alpha), 1.0f },
                           { P[0] + 10.0f * cosf(alpha), P[1] - 10.0f * sinf(alpha), 1.0f } };

  // the two diagonals: D = P x Pa
  float D[2][3];
  vec3prodn(D[0], P, Pa[0]);
  vec3prodn(D[1], P, Pa[1]);

  // find all intersection points of all four edges with both diagonals (I = E x D);
  // the shortest distance d2min of the intersection point I to the crop area center P determines
  // the size of the crop area that still fits into the image (for the given center and aspect angle)
  float d2min = FLT_MAX;
  for(int k = 0; k < 4; k++)
    for(int l = 0; l < 2; l++)
    {
      // the intersection point
      float I[3];
      vec3prodn(I, cropfit->edges[k], D[l]);

      // special case: I is all null -> E and D are identical -> P lies on E -> d2min = 0
      if(vec3isnull(I))
      {
        d2min = 0.0f;
        break;
      }

      // special case: I[2] is 0.0f -> E and D are parallel and intersect at infinity -> no relevant point
      if(I[2] == 0.0f)
        continue;

      // the default case -> normalize I
      I[0] /= I[2];
      I[1] /= I[2];

      // calculate distance from I to P
      const float d2 = SQR(P[0] - I[0]) + SQR(P[1] - I[1]);

      // the minimum distance over all intersection points
      d2min = MIN(d2min, d2);
    }

  // calculate the area of the rectangle
  const float A = 2.0f * d2min * sinf(2.0f * alpha);

#ifdef ASHIFT_DEBUG
  printf("crop fitness with x %f, y %f, angle %f -> distance %f, area %f\n",
         x, y, alpha, d2min, A);
#endif
  // and return -A to allow Nelder-Mead simplex to search for the minimum
  return -A;
}

float * shift(
    float width, float height,
    int input_line_count,
    rect rects[]
) {
    printf("Shift\n");
    printf("Width: %f\n", width);
    printf("Height: %f\n", height);
    printf("Input Line Count: %i\n", input_line_count);

    dt_iop_ashift_line_t *lines;
    int lines_count;
    int vertical_count;
    int horizontal_count;
    float vertical_weight;
    float horizontal_weight;

   printf("Line Process: %i\n", line_prcoess(
        input_line_count,
        rects,
        width, height,
        0.0f, 0.0f, // TODO: Work out what these parameters do
        1.0f, // TODO: Work out what these parameters do
        &lines, &lines_count,
        &vertical_count, &horizontal_count, &vertical_weight, &horizontal_weight
    ));

    dt_iop_ashift_gui_data_t g;

    g.rotation_range = ROTATION_RANGE_SOFT;
    g.lensshift_v_range = LENSSHIFT_RANGE_SOFT;
    g.lensshift_h_range = LENSSHIFT_RANGE_SOFT;
    g.shear_range = SHEAR_RANGE_SOFT;
    g.lines_in_width = width;
    g.lines_in_height = height;
    g.lines_x_off = 0.0f; // TODO: Work out what these parameters do
    g.lines_y_off = 0.0f; // TODO: Work out what these parameters do
    g.lines = lines;
    g.lines_count = lines_count;
    g.buf_width = width;
    g.buf_height = height;

    printf("Processed Line Count: %i\n", lines_count);
    printf("Proccessed Outliers: %i\n", _remove_outliers(&g));
    printf("Outlier Line Count: %i\n", lines_count);

    dt_iop_ashift_params_t p;

    p.rotation = 0;
    p.lensshift_v = 0;
    p.lensshift_h = 0;
    p.shear = 0;
    p.mode = ASHIFT_MODE_GENERIC;
    p.cl = 0.0;
    p.cr = 1.0;
    p.ct = 0.0;
    p.cb = 1.0;

    dt_iop_ashift_fitaxis_t dir = ASHIFT_FIT_VERTICALLY_NO_ROTATION;
    dt_iop_ashift_nmsresult_t res = nmsfit(&g, &p, dir);

    switch(res)
    {
      case NMS_NOT_ENOUGH_LINES:
        printf("Not enough structure for automatic correction\nminimum %d lines in each relevant direction\n",
            MINIMUM_FITLINES);
        error("Failed: Not enough lines");
        break;
      case NMS_DID_NOT_CONVERGE:
      case NMS_INSANE:
        printf("automatic correction failed, please correct manually\n");
        error("Failed: No convergence or insane");
        break;
      case NMS_SUCCESS:
      default:
        printf("Successfully adjusted\n");
        break;
    }

    printf("R: %f\n", p.rotation);
    printf("LV: %f\n", p.lensshift_v);
    printf("LH: %f\n", p.lensshift_h);
    printf("S: %f\n", p.shear);

    float homograph[3][3];
    homography((float *)homograph, p.rotation, p.lensshift_v, p.lensshift_h, p.shear, DEFAULT_F_LENGTH,
              100, 1.0, width, height, ASHIFT_HOMOGRAPH_FORWARD);

    printf("[[ %f,", homograph[0][0]);
    printf("%f,", homograph[0][1]);
    printf("%f],\n", homograph[0][2]);
    printf("[%f,", homograph[1][0]);
    printf("%f,", homograph[1][1]);
    printf("%f],\n", homograph[1][2]);
    printf("[%f,", homograph[2][0]);
    printf("%f,", homograph[2][1]);
    printf("%f]]\n", homograph[2][2]);

    float *flatMatrix = malloc(sizeof(float) * 9);

    flatMatrix[0] = homograph[0][0];
    flatMatrix[1] = homograph[0][1];
    flatMatrix[2] = homograph[0][2];
    flatMatrix[3] = homograph[1][0];
    flatMatrix[4] = homograph[1][1];
    flatMatrix[5] = homograph[1][2];
    flatMatrix[6] = homograph[2][0];
    flatMatrix[7] = homograph[2][1];
    flatMatrix[8] = homograph[2][2]; 

    return flatMatrix;
  };