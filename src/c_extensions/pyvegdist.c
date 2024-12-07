#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

// Chord distance function
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

// Canberra distance function
static double veg_canberra(double *x, int nr, int nc, int i1, int i2) {
    double dist = 0.0;
    int count = 0;
    double epsilon = 1e-12;  // 阈值，避免分母接近零的情况

    for (int j = 0; j < nc; j++) {
        double x1 = x[i1 * nc + j];  // 访问 i1 行 j 列
        double x2 = x[i2 * nc + j];  // 访问 i2 行 j 列

        // 检查是否为 NaN 值
        if (!isnan(x1) && !isnan(x2)) {
            // 检查 x1 和 x2 是否同时为 0
            if (x1 != 0 || x2 != 0) {
                double numer = fabs(x1 - x2);
                double denom = fabs(x1) + fabs(x2);

                // 检查分母是否大于阈值，避免分母接近零的情况
                if (denom > epsilon) {
                    dist += numer / denom;
                    count++;  // 只有当分母有效时才增加 count
                }
            }
        }
    }

    // 如果没有任何有效的计算，返回 NaN
    if (count == 0) return NAN;
    return dist / (double)count;  // 返回平均距离
}

// Canberra distance function
// static double veg_canberra(double *x, int nr, int nc, int i1, int i2) {
//     double dist = 0.0;
//     int count = 0;

//     for (int j = 0; j < nc; j++) {
//         if (!isnan(x[i1 * nc + j]) && !isnan(x[i2 * nc + j])) {
//             double numer = fabs(x[i1 * nc + j] - x[i2 * nc + j]);
//             double denom = fabs(x[i1 * nc + j]) + fabs(x[i2 * nc + j]);
//             if (denom > 0.0) {
//                 dist += numer / denom;
//             } else {
//                 dist += INFINITY;
//             }
//             count++;
//         }
//     }
//     if (count == 0) return NAN;
//     return dist / (double)count;
// }



// Manhattan distance function
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

// Euclidean distance function
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

// Compute distance matrix
static PyObject* compute_distance_matrix(PyObject* self, PyObject* args) {
    const char* method;
    PyObject* x_obj;
    PyArrayObject* x;
    double (*distance_func)(double*, int, int, int, int) = NULL;

    // Parse Python arguments
    if (!PyArg_ParseTuple(args, "sO", &method, &x_obj)) {
        return NULL;
    }

    // Convert Python object to numpy array
    x = (PyArrayObject*)PyArray_FROMANY(x_obj, NPY_DOUBLE, 2, 2, NPY_ARRAY_IN_ARRAY);
    if (x == NULL) {
        return NULL;
    }

    int nr = PyArray_DIM(x, 0);
    int nc = PyArray_DIM(x, 1);
    double* x_data = (double*)PyArray_DATA(x);

    // Choose the appropriate distance function
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

    // Allocate memory for the distance matrix
    npy_intp dims[2] = { nr, nr };
    PyArrayObject* d = (PyArrayObject*)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (d == NULL) {
        Py_DECREF(x);
        return NULL;
    }
    double* d_data = (double*)PyArray_DATA(d);

    // Compute the distance matrix
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

// Method definitions
static PyMethodDef DistanceMethods[] = {
    {"compute_distance_matrix", compute_distance_matrix, METH_VARARGS, "Compute distance matrix"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition
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
