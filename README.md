# PW2

## Exercise 1 – Image processing methods

### 1 - Anaglyph color methods (on image.jpg)
parameters: method (1-5)
- anaglyph.cpp --> anaglyph.o --> anaglyph
- anaglyphCUDA.cpp + anaglyphcuda.cu --> anaglyphcuda

### 2 – Gaussian filtering (on image.jpg)
parameters: kernel size, sigma
- gaussian.cpp --> gaussian.o --> gaussian
- gaussianCUDA.cpp + gaussiancuda.cu --> gaussiancuda

### 3 – Denoising (on painting.tif)
parameters: neighborhood size, ratio
- denoising.cpp --> denoising.o --> denoising
- denoisingCUDA.cpp + denoisingcuda.cu --> denoisingcuda

## Exercise 2 – Shared memory optimization (on image.jpg)
- gaussianCUDA.cpp + gaussiancudashared.cu --> gaussiancudashared

## Exercises 3 – Execution time comparisons
- PW2-benchmark.xlsx
