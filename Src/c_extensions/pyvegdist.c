/**
 * @file pyvegdist.c
 * @brief A Python extension module for computing distance matrices using various methods.
 *
 * This module is derived from the vegan R package (GPL-2 licensed) and provides
 * functionality to calculate distance matrices for ecological community data.
 *
 * The original vegan package is part of the vegan collection of the R package series,
 * which is a toolbox for community ecologists.
 *
 * Copyright (c) 2008-2019 Jari Oksanen and F. Guillaume Blanchet
 * Original R package: https://CRAN.R-project.org/package=vegan
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

/* Chord & Hellinger distances can be calculated after appropriate
* standardization in veg_euclidean, but it is also possible to use
* this function for non-standardized (Chord) data or square root
* transformed (Hellinger) data. It may be that veg_euclidean() is
* numerically more stable: the current function collects sum of
* squares and cross-products which can have loss of precision
*/
static double veg_chord(double *x, int nr, int nc, int i1, int i2) {
    double cp = 0.0, ss1 = 0.0, ss2 = 0.0;
    int count = 0;

    for (int j = 0; j < nc; j++) {
        if (!isnan(x[i1 * nc + j]) && !isnan(x[i2 * nc + j])) {
            count++;
            cp += x[i1 * nc + j] * x[i2 * nc + j];
            ss1 += x[i1 * nc + j] * x[i1 * nc + j];
            ss2 += x[i2 * nc + j] * x[i2 * nc + j];
        }
    }

    if (count == 0) return NAN;
    double dist = 2.0 * (1.0 - cp / sqrt(ss1 * ss2));
    return sqrt(dist);
}

/* Canberra distance: duplicates R base, but is scaled into range
 * 0...1  
*/

static double veg_canberra(double *x, int nr, int nc, int i1, int i2) {
    double dist = 0.0;
    int count = 0;
    double epsilon = 1e-12;

    for (int j = 0; j < nc; j++) {
        double x1 = x[i1 * nc + j];
        double x2 = x[i2 * nc + j];

        if (!isnan(x1) && !isnan(x2)) {
            if (x1 != 0 || x2 != 0) {
                double numer = fabs(x1 - x2);
                double denom = fabs(x1) + fabs(x2);

                if (denom > epsilon) {
                    dist += numer / denom;
                    count++;
                }
            }
        }
    }

    if (count == 0) return NAN;
    return dist / (double)count;
}

/* Manhattan distance: duplicates base R */
static double veg_manhattan(double *x, int nr, int nc, int i1, int i2) {
    double dist = 0.0;
    int count = 0;

    for (int j = 0; j < nc; j++) {
        if (!isnan(x[i1 * nc + j]) && !isnan(x[i2 * nc + j])) {
            dist += fabs(x[i1 * nc + j] - x[i2 * nc + j]);
            count++;
        }
    }
    if (count == 0) return NAN;
    return dist;
}

/* Euclidean distance: duplicates base R. This function can be (and will be)
* used for many named distance indices with transforming & standardizing input
* data in the calling R code (vegdist): Chord, Hellinger, Chi-square,
* Aitchison, Mahalanobis -- see veg_distance() below for actual list at the
* moment.
*/
static double veg_euclidean(double *x, int nr, int nc, int i1, int i2) {
    double dist = 0.0;
    int count = 0;

    for (int j = 0; j < nc; j++) {
        if (!isnan(x[i1 * nc + j]) && !isnan(x[i2 * nc + j])) {
            double dev = x[i1 * nc + j] - x[i2 * nc + j];
            dist += dev * dev;
            count++;
        }
    }
    if (count == 0) return NAN;
    return sqrt(dist);
}


static PyObject* compute_distance_matrix(PyObject* self, PyObject* args) {
    const char* method;
    PyObject* x_obj;
    PyArrayObject* x;
    double (*distance_func)(double*, int, int, int, int) = NULL;

    if (!PyArg_ParseTuple(args, "sO", &method, &x_obj)) {
        return NULL;
    }
    x = (PyArrayObject*)PyArray_FROMANY(x_obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (x == NULL) {
        return NULL;
    }

    int nr = PyArray_DIM(x, 0);
    int nc = PyArray_DIM(x, 1);
    double* x_data = (double*)PyArray_DATA(x);
    if (strcmp(method, "chord") == 0) {
        distance_func = veg_chord;
    } else if (strcmp(method, "canberra") == 0) {
        distance_func = veg_canberra;
    } else if (strcmp(method, "manhattan") == 0) {
        distance_func = veg_manhattan;
    } else if (strcmp(method, "euclidean") == 0) {
        distance_func = veg_euclidean;
    } else {
        Py_DECREF(x);
        PyErr_SetString(PyExc_ValueError, "Unknown distance method");
        return NULL;
    }
    npy_intp dims[2] = { nr, nr };
    PyArrayObject* d = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (d == NULL) {
        Py_DECREF(x);
        return NULL;
    }
    double* d_data = (double*)PyArray_DATA(d);

    for (int i = 0; i < nr; i++) {
        d_data[i * nr + i] = 0.0;  // Diagonal is zero
        for (int j = i + 1; j < nr; j++) {
            double dist = distance_func(x_data, nr, nc, i, j);
            d_data[i * nr + j] = dist;
            d_data[j * nr + i] = dist;  // Symmetry
        }
    }

    Py_DECREF(x);
    return (PyObject*)d;
}

static PyMethodDef DistanceMethods[] = {
    {"compute_distance_matrix", compute_distance_matrix, METH_VARARGS, "Compute distance matrix"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

static struct PyModuleDef distancemodule = {
    PyModuleDef_HEAD_INIT,
    "pyvegdist",
    NULL,
    -1,
    DistanceMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_pyvegdist(void) {
    import_array();  // For numpy
    return PyModule_Create(&distancemodule);
}
