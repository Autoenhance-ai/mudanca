
// simple conversion of rgb image into greyscale variant suitable for line segment detection
// the lsd routines expect input as *double, roughly in the range [0.0; 256.0]
static void rgb2grey256(const float *const in, double *const out, const int width, const int height)
{
  const size_t npixels = (size_t)width * height;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(npixels) \
  dt_omp_sharedconst(in, out) \
  schedule(static)
#endif
  for(int index = 0; index < npixels; index++)
  {
    out[index] = (0.3f * in[4*index+0] + 0.59f * in[4*index+1] + 0.11f * in[4*index+2]) * 256.0;
  }
}

// sobel edge enhancement in one direction
static void edge_enhance_1d(const double *in, double *out, const int width, const int height,
                            dt_iop_ashift_enhance_t dir)
{
  // Sobel kernels for both directions
  const double hkernel[3][3] = { { 1.0, 0.0, -1.0 }, { 2.0, 0.0, -2.0 }, { 1.0, 0.0, -1.0 } };
  const double vkernel[3][3] = { { 1.0, 2.0, 1.0 }, { 0.0, 0.0, 0.0 }, { -1.0, -2.0, -1.0 } };
  const int kwidth = 3;
  const int khwidth = kwidth / 2;

  // select kernel
  const double *kernel = (dir == ASHIFT_ENHANCE_HORIZONTAL) ? (const double *)hkernel : (const double *)vkernel;

#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(height, width, khwidth, kwidth) \
  shared(in, out, kernel) \
  schedule(static)
#endif
  // loop over image pixels and perform sobel convolution
  for(int j = khwidth; j < height - khwidth; j++)
  {
    const double *inp = in + (size_t)j * width + khwidth;
    double *outp = out + (size_t)j * width + khwidth;
    for(int i = khwidth; i < width - khwidth; i++, inp++, outp++)
    {
      double sum = 0.0f;
      for(int jj = 0; jj < kwidth; jj++)
      {
        const int k = jj * kwidth;
        const int l = (jj - khwidth) * width;
        for(int ii = 0; ii < kwidth; ii++)
        {
          sum += inp[l + ii - khwidth] * kernel[k + ii];
        }
      }
      *outp = sum;
    }
  }

#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(height, width, khwidth) \
  shared(out) \
  schedule(static)
#endif
  // border fill in output buffer, so we don't get pseudo lines at image frame
  for(int j = 0; j < height; j++)
    for(int i = 0; i < width; i++)
    {
      double val = out[j * width + i];

      if(j < khwidth)
        val = out[(khwidth - j) * width + i];
      else if(j >= height - khwidth)
        val = out[(j - khwidth) * width + i];
      else if(i < khwidth)
        val = out[j * width + (khwidth - i)];
      else if(i >= width - khwidth)
        val = out[j * width + (i - khwidth)];

      out[j * width + i] = val;

      // jump over center of image
      if(i == khwidth && j >= khwidth && j < height - khwidth) i = width - khwidth;
    }
}

// edge enhancement in both directions
static int edge_enhance(const double *in, double *out, const int width, const int height)
{
  double *Gx = NULL;
  double *Gy = NULL;

  Gx = malloc(sizeof(double) * width * height);
  if(Gx == NULL) goto error;

  Gy = malloc(sizeof(double) * width * height);
  if(Gy == NULL) goto error;

  // perform edge enhancement in both directions
  edge_enhance_1d(in, Gx, width, height, ASHIFT_ENHANCE_HORIZONTAL);
  edge_enhance_1d(in, Gy, width, height, ASHIFT_ENHANCE_VERTICAL);

// calculate absolute values
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(height, width) \
  shared(Gx, Gy, out) \
  schedule(static)
#endif
  for(size_t k = 0; k < (size_t)width * height; k++)
  {
    out[k] = sqrt(Gx[k] * Gx[k] + Gy[k] * Gy[k]);
  }

  free(Gx);
  free(Gy);
  return TRUE;

error:
  if(Gx) free(Gx);
  if(Gy) free(Gy);
  return FALSE;
}

// XYZ -> sRGB matrix
static void XYZ_to_sRGB(const dt_aligned_pixel_t XYZ, dt_aligned_pixel_t sRGB)
{
  sRGB[0] =  3.1338561f * XYZ[0] - 1.6168667f * XYZ[1] - 0.4906146f * XYZ[2];
  sRGB[1] = -0.9787684f * XYZ[0] + 1.9161415f * XYZ[1] + 0.0334540f * XYZ[2];
  sRGB[2] =  0.0719453f * XYZ[0] - 0.2289914f * XYZ[1] + 1.4052427f * XYZ[2];
}

// sRGB -> XYZ matrix
static void sRGB_to_XYZ(const dt_aligned_pixel_t sRGB, dt_aligned_pixel_t XYZ)
{
  XYZ[0] = 0.4360747f * sRGB[0] + 0.3850649f * sRGB[1] + 0.1430804f * sRGB[2];
  XYZ[1] = 0.2225045f * sRGB[0] + 0.7168786f * sRGB[1] + 0.0606169f * sRGB[2];
  XYZ[2] = 0.0139322f * sRGB[0] + 0.0971045f * sRGB[1] + 0.7141733f * sRGB[2];
}

// detail enhancement via bilateral grid (function arguments in and out may represent identical buffers)
static int detail_enhance(const float *const in, float *const out, const int width, const int height)
{
  const float sigma_r = 5.0f;
  const float sigma_s = fminf(width, height) * 0.02f;
  const float detail = 10.0f;
  const size_t npixels = (size_t)width * height;
  int success = TRUE;

  // we need to convert from RGB to Lab first;
  // as colors don't matter we are safe to assume data to be sRGB

  // convert RGB input to Lab, use output buffer for intermediate storage
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(npixels) \
  dt_omp_sharedconst(in, out) \
  schedule(static)
#endif
  for(size_t index = 0; index < 4*npixels; index += 4)
  {
    dt_aligned_pixel_t XYZ;
    sRGB_to_XYZ(in + index, XYZ);
    dt_XYZ_to_Lab(XYZ, out + index);
  }

  // bilateral grid detail enhancement
  dt_bilateral_t *b = dt_bilateral_init(width, height, sigma_s, sigma_r);

  if(b != NULL)
  {
    dt_bilateral_splat(b, out);
    dt_bilateral_blur(b);
    dt_bilateral_slice_to_output(b, out, out, detail);
    dt_bilateral_free(b);
  }
  else
    success = FALSE;

  // convert resulting Lab to RGB output
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(npixels) \
  dt_omp_sharedconst(out) \
  schedule(static)
#endif
  for(size_t index = 0; index < 4*npixels; index += 4)
  {
    dt_aligned_pixel_t XYZ;
    dt_Lab_to_XYZ(out + index, XYZ);
    XYZ_to_sRGB(XYZ, out + index);
  }

  return success;
}

// apply gamma correction to RGB buffer (function arguments in and out may represent identical buffers)
static void gamma_correct(const float *const in, float *const out, const int width, const int height)
{
  const size_t npixels = (size_t)width * height;
#ifdef _OPENMP
#pragma omp parallel for default(none) \
  dt_omp_firstprivate(npixels) \
  dt_omp_sharedconst(in, out) \
  schedule(static)
#endif
  for(int index = 0; index < 4*npixels; index += 4)
  {
    for(int c = 0; c < 3; c++)
      out[index+c] = powf(in[index+c], LSD_GAMMA);
  }
}


// do actual line_detection based on LSD algorithm and return results according
// to this module's conventions
static int line_detect(float *in, const int width, const int height, const int x_off, const int y_off,
                       const float scale, dt_iop_ashift_line_t **alines, int *lcount, int *vcount, int *hcount,
                       float *vweight, float *hweight, dt_iop_ashift_enhance_t enhance, const int is_raw)
{
  double *greyscale = NULL;
  double *lsd_lines = NULL;
  dt_iop_ashift_line_t *ashift_lines = NULL;

  int vertical_count = 0;
  int horizontal_count = 0;
  float vertical_weight = 0.0f;
  float horizontal_weight = 0.0f;

  // apply gamma correction if image is raw
  if(is_raw)
  {
    gamma_correct(in, in, width, height);
  }

  // if requested perform an additional detail enhancement step
  if(enhance & ASHIFT_ENHANCE_DETAIL)
  {
    (void)detail_enhance(in, in, width, height);
  }

  // allocate intermediate buffers
  greyscale = malloc(sizeof(double) * width * height);
  if(greyscale == NULL) goto error;

  // convert to greyscale image
  rgb2grey256(in, greyscale, width, height);

  // if requested perform an additional edge enhancement step
  if(enhance & ASHIFT_ENHANCE_EDGES)
  {
    (void)edge_enhance(greyscale, greyscale, width, height);
  }

  // call the line segment detector LSD;
  // LSD stores the number of found lines in lines_count.
  // it returns structural details as vector 'double lines[7 * lines_count]'
  int lines_count;

  lsd_lines = LineSegmentDetection(&lines_count, greyscale, width, height,
                                   LSD_SCALE, LSD_SIGMA_SCALE, LSD_QUANT,
                                   LSD_ANG_TH, LSD_LOG_EPS, LSD_DENSITY_TH,
                                   LSD_N_BINS, NULL, NULL, NULL);

  // we count the lines that we really want to use
  int lct = 0;
  if(lines_count > 0)
  {
    // aggregate lines data into our own structures
    ashift_lines = (dt_iop_ashift_line_t *)malloc(sizeof(dt_iop_ashift_line_t) * lines_count);
    if(ashift_lines == NULL) goto error;

    for(int n = 0; n < lines_count; n++)
    {
      const float x1 = lsd_lines[n * 7 + 0];
      const float y1 = lsd_lines[n * 7 + 1];
      const float x2 = lsd_lines[n * 7 + 2];
      const float y2 = lsd_lines[n * 7 + 3];

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
      ashift_lines[lct].width = lsd_lines[n * 7 + 4] / scale;

      // ...  and weight (= length * width * angle precision)
      const float weight = ashift_lines[lct].length * ashift_lines[lct].width * lsd_lines[n * 7 + 5];
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
  free(lsd_lines);
  free(greyscale);
  return lct > 0 ? TRUE : FALSE;

error:
  free(lsd_lines);
  free(greyscale);
  return FALSE;
}
