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

#define shift_ALIGNED_ARRAY __attribute__((aligned(64)))

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

#define ROTATION_RANGE 5                    // allowed min/max default range for rotation parameter
#define LENSSHIFT_RANGE 2.0                 // allowed min/max default range for lensshift parameters
#define SHEAR_RANGE 0.5                     // allowed min/max range for shear parameter
#define MIN_LINE_LENGTH 5                   // the minimum length of a line in pixels to be regarded as relevant
#define MAX_TANGENTIAL_DEVIATION 10         // by how many degrees a line may deviate from the +/-180 and +/-90 to be regarded as relevant
#define MINIMUM_FITLINES 2                  // minimum number of lines needed for automatic parameter fit
#define NMS_EPSILON 1e-3                    // break criterion for Nelder-Mead simplex
#define NMS_SCALE 1.0                       // scaling factor for Nelder-Mead simplex
#define NMS_ITERATIONS 400                  // number of iterations for Nelder-Mead simplex
#define NMS_ALPHA 1.0                       // reflection coefficient for Nelder-Mead simplex
#define NMS_BETA 0.5                        // contraction coefficient for Nelder-Mead simplex
#define NMS_GAMMA 2.0                       // expansion coefficient for Nelder-Mead simplex
#define DEFAULT_F_LENGTH 28.0               // focal length we assume if no exif data are available

// define to get debugging output
// #define ASHIFT_DEBUG

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

typedef enum shift_iop_ashift_homodir_t
{
  ASHIFT_HOMOGRAPH_FORWARD,
  ASHIFT_HOMOGRAPH_INVERTED
} shift_iop_ashift_homodir_t;

typedef enum shift_iop_ashift_linetype_t
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
} shift_iop_ashift_linetype_t;

typedef enum shift_iop_ashift_fitaxis_t
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
} shift_iop_ashift_fitaxis_t;

typedef enum shift_iop_ashift_nmsresult_t
{
  NMS_SUCCESS = 0,
  NMS_NOT_ENOUGH_LINES = 1,
  NMS_DID_NOT_CONVERGE = 2,
  NMS_INSANE = 3
} shift_iop_ashift_nmsresult_t;

typedef enum shift_iop_ashift_mode_t
{
  ASHIFT_MODE_GENERIC = 0, // $DESCRIPTION: "generic"
  ASHIFT_MODE_SPECIFIC = 1 // $DESCRIPTION: "specific"
} shift_iop_ashift_mode_t;

typedef struct shift_iop_ashift_params_t
{
  float rotation;    // $MIN: -ROTATION_RANGE_SOFT $MAX: ROTATION_RANGE_SOFT $DEFAULT: 0.0
  float lensshift_v; // $MIN: -LENSSHIFT_RANGE_SOFT $MAX: LENSSHIFT_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "lens shift (vertical)"
  float lensshift_h; // $MIN: -LENSSHIFT_RANGE_SOFT $MAX: LENSSHIFT_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "lens shift (horizontal)"
  float shear;       // $MIN: -SHEAR_RANGE_SOFT $MAX: SHEAR_RANGE_SOFT $DEFAULT: 0.0 $DESCRIPTION: "shear"
  float f_length;    // $MIN: 1.0 $MAX: 2000.0 $DEFAULT: DEFAULT_F_LENGTH $DESCRIPTION: "focal length"
  float crop_factor; // $MIN: 0.5 $MAX: 10.0 $DEFAULT: 1.0 $DESCRIPTION: "crop factor"
  float orthocorr;   // $MIN: 0.0 $MAX: 100.0 $DEFAULT: 100.0 $DESCRIPTION: "lens dependence"
  float aspect;      // $MIN: 0.5 $MAX: 2.0 $DEFAULT: 1.0 $DESCRIPTION: "aspect adjust"
  shift_iop_ashift_mode_t mode;     // $DEFAULT: ASHIFT_MODE_GENERIC $DESCRIPTION: "lens model"
  float cl;          // $DEFAULT: 0.0
  float cr;          // $DEFAULT: 1.0
  float ct;          // $DEFAULT: 0.0
  float cb;          // $DEFAULT: 1.0
} shift_iop_ashift_params_t;

typedef struct shift_iop_ashift_line_t
{
  float p1[3];
  float p2[3];
  float length;
  float width;
  float weight;
  shift_iop_ashift_linetype_t type;
  // homogeneous coordinates:
  float L[3];
} shift_iop_ashift_line_t;

typedef struct shift_iop_ashift_fit_params_t
{
  int params_count;
  shift_iop_ashift_linetype_t linetype;
  shift_iop_ashift_linetype_t linemask;
  shift_iop_ashift_line_t *lines;
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
} shift_iop_ashift_fit_params_t;

typedef struct shift_iop_ashift_cropfit_params_t
{
  int width;
  int height;
  float x;
  float y;
  float alpha;
  float homograph[3][3];
  float edges[4][3];
} shift_iop_ashift_cropfit_params_t;

typedef struct shift_iop_ashift_gui_data_t
{
  float rotation_range;
  float lensshift_v_range;
  float lensshift_h_range;
  float shear_range;
  shift_iop_ashift_line_t *lines;
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
  float cl;	// shadow copy of shift_iop_ashift_data_t.cl
  float cr;	// shadow copy of shift_iop_ashift_data_t.cr
  float ct;	// shadow copy of shift_iop_ashift_data_t.ct
  float cb;	// shadow copy of shift_iop_ashift_data_t.cb
} shift_iop_ashift_gui_data_t;

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

// process detectedlines according
// to this module's conventions
static int line_prcoess(
    const int lines_count,
    rect rects[],
    const int width, const int height,
    const int x_off, const int y_off,
    const float scale,
    shift_iop_ashift_line_t **alines,
    int *lcount, int *vcount, int *hcount,
    float *vweight, float *hweight
) {

  shift_iop_ashift_line_t *ashift_lines = NULL;

  int vertical_count = 0;
  int horizontal_count = 0;
  float vertical_weight = 0.0f;
  float horizontal_weight = 0.0f;

  // we count the lines that we really want to use
  int lct = 0;
  if(lines_count > 0)
  {
    // aggregate lines data into our own structures
    ashift_lines = (shift_iop_ashift_line_t *)malloc(sizeof(shift_iop_ashift_line_t) * lines_count);
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
      const float weight = ashift_lines[lct].length * ashift_lines[lct].width * rects[n].precision;
      ashift_lines[lct].weight = weight; 

      const float angle = atan2f(py2 - py1, px2 - px1) / M_PI * 180.0f;
      const int vertical = fabsf(fabsf(angle) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;
      const int horizontal = fabsf(fabsf(fabsf(angle) - 90.0f) - 90.0f) < MAX_TANGENTIAL_DEVIATION ? 1 : 0;

      const int relevant = ashift_lines[lct].length > MIN_LINE_LENGTH ? 1 : 0;

      // register type of line
      shift_iop_ashift_linetype_t type = ASHIFT_LINE_IRRELEVANT;
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
                       const int width, const int height, shift_iop_ashift_homodir_t dir)
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
  shift_iop_ashift_fit_params_t *fit = (shift_iop_ashift_fit_params_t *)data;

  // just for convenience: get shorter names
  shift_iop_ashift_line_t *lines = fit->lines;
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
static shift_iop_ashift_nmsresult_t nmsfit(shift_iop_ashift_gui_data_t *g, shift_iop_ashift_params_t *p, shift_iop_ashift_fitaxis_t dir)
{
  if(!g->lines) return NMS_NOT_ENOUGH_LINES;
  if(dir == ASHIFT_FIT_NONE) return NMS_SUCCESS;

  double params[4];
  int pcount = 0;
  int enough_lines = TRUE;

  // initialize fit parameters
  shift_iop_ashift_fit_params_t fit;
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
  shift_iop_ashift_fitaxis_t mdir = dir;

  // prepare fit structure and starting parameters for simplex fit.
  // note: the sequence of parameters in params[] needs to match the
  // respective order in shift_iop_ashift_fit_params_t. Parameters which are
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

float * shift(
    float width, float height,
    int input_line_count,
    rect rects[],
    int options
) {

    shift_iop_ashift_line_t *lines;
    int lines_count;
    int vertical_count;
    int horizontal_count;
    float vertical_weight;
    float horizontal_weight;

    line_prcoess(
        input_line_count,
        rects,
        width, height,
        0.0f, 0.0f,
        1.0f,
        &lines, &lines_count,
        &vertical_count, &horizontal_count, &vertical_weight, &horizontal_weight
    );

    shift_iop_ashift_gui_data_t g;

    g.rotation_range = ROTATION_RANGE;
    g.lensshift_v_range = LENSSHIFT_RANGE;
    g.lensshift_h_range = LENSSHIFT_RANGE;
    g.shear_range = SHEAR_RANGE;
    g.lines_in_width = width;
    g.lines_in_height = height;
    g.lines_x_off = 0.0f;
    g.lines_y_off = 0.0f;
    g.lines = lines;
    g.lines_count = lines_count;
    g.buf_width = width;
    g.buf_height = height;
    
    g.vertical_weight = vertical_weight;
    g.horizontal_weight =horizontal_weight;
    g.vertical_count = vertical_count;
    g.horizontal_count = horizontal_count;

    shift_iop_ashift_params_t p;

    p.rotation = 0;
    p.lensshift_v = 0;
    p.lensshift_h = 0;
    p.shear = 0;
    p.mode = ASHIFT_MODE_GENERIC;
    p.aspect = 1.0f;
    p.orthocorr = 0.0f;
    p.cl = 0.0;
    p.cr = 1.0;
    p.ct = 0.0;
    p.cb = 1.0;

    shift_iop_ashift_fitaxis_t dir = options;
    shift_iop_ashift_nmsresult_t res = nmsfit(&g, &p, dir);

    switch(res)
    {
      case NMS_NOT_ENOUGH_LINES:
        printf("Not enough structure for automatic correction\nminimum %d lines in each relevant direction\n",
            MINIMUM_FITLINES);
        printf("Failed: Not enough lines");
        break;
      case NMS_DID_NOT_CONVERGE:
      case NMS_INSANE:
        printf("automatic correction failed, please correct manually\n");
        printf("Failed: No convergence or insane");
        break;
      case NMS_SUCCESS:
      default:
        break;
    }

    float homograph[3][3];
    homography((float *)homograph, p.rotation, p.lensshift_v, p.lensshift_h, p.shear, DEFAULT_F_LENGTH,
              p.orthocorr, p.aspect, width, height, ASHIFT_HOMOGRAPH_FORWARD);

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

    #ifdef ASHIFT_DEBUG
    printf("ROT: %f\n", p.rotation);
    printf("LV: %f\n", p.lensshift_v);
    printf("LH: %f\n", p.lensshift_h);
    printf("SHR: %f\n", p.shear);
    #endif

    return flatMatrix;
  };