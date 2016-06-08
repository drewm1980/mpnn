#include <cstdlib>
#include <cstdio>  // C I/O
#include <cstdint>
#include <fstream>   // file I/O
#include <iostream>  // file I/O
#include <string>    // string manipulation
#include <math.h>    // math routines
#include <list>
#include <queue>
#include <time.h>

#include <ANN.h>  // ANN declarations
#include <multiann.h>

using namespace std;

#ifndef INFINITY
#define INFINITY 1.0e40
#endif
#ifndef PI
#define PI 3.1415926535897932385
#endif

// dim: dimension of the embedding of the product space, i.e. 5 for S1 x RP3
// tt: individual topologies in the cross product: 
// 2 for -PI to PI
// 3 for quaternion
// 1 for -100 to 100
// p: the point in the product space
//
// Doesn't do any memory allocation; p must be large enough!
void randomDistPoint(int dim, const int *tt, ANNpoint &p) {
    for (int kk = 0; kk < dim; kk++) {
        double tmp = rand() / (2.14783e+09);
        if (tt[kk] == 2) {
            p[kk] = (2 * PI * tmp - PI);  // region -PI PI
        } else if (tt[kk] == 3) {
            double tmp1 = rand() / (2.14783e+09);
            double tmp2 = rand() / (2.14783e+09);

            double Theta1 = 2 * PI * tmp1;
            double Theta2 = PI * tmp2 - PI / 2;
            double r1 = sqrt(1 - tmp);
            double r2 = sqrt(tmp);

            p[kk] = sin(Theta1) * r1;
            p[kk + 1] = cos(Theta1) * r1;
            p[kk + 2] = sin(Theta2) * r2;
            p[kk + 3] = cos(Theta2) * r2;

            // Modifies loop index!
            kk = kk + 3;  // random quaternions
        } else
            p[kk] = 200 * tmp - 100;  // region -100 to 100
    }
}

// Generate an array of random points.
// Does no allocation; data_pts must be large enough!
void randomDist(int dim, const int *tt, ANNpointArray &data_pts, int m_pts) {
    int n_pts = 0;
    while (n_pts < m_pts) {
        randomDistPoint(dim, tt, data_pts[n_pts]);
        n_pts++;
    }
}

// Compute the distance metric between two points.
// Default is to use the standard Euclidean metric (defined by #defines in
// ANN.h)
// x1,x2: the two points in the space
// dim: the dimension of the embedding of the product space.
// topology: array of codes for the individual topologies
double Metric(const ANNpoint x1, const ANNpoint x2, int dim, int *topology,
              ANNpoint scale) {
    double rho = 0, fd, dtheta;

    for (int i = 0; i < dim; i++) {
        if (topology[i] == 1) {
            rho += ANN_POW(scale[i] * (x1[i] - x2[i]));
        } else if (topology[i] == 2) {
            fd = fabs(x1[i] - x2[i]);
            dtheta = ANN_MIN(fd, 2.0 * PI - fd);
            rho += ANN_POW(scale[i] * dtheta);
        } else if (topology[i] == 3) {
            // dot product
            fd = x1[i] * x2[i] + x1[i + 1] * x2[i + 1] + x1[i + 2] * x2[i + 2] +
                 x1[i + 3] * x2[i + 3];
            // Handle non-unit quaternions only if they're larger than 1 (why?)
            // should this be >0 to avoid divide by zero?
            if (fd > 1) {
                double norm1 = x1[i] * x1[i] + x1[i + 1] * x1[i + 1] +
                               x1[i + 2] * x1[i + 2] + x1[i + 3] * x1[i + 3];
                double norm2 = x2[i] * x2[i] + x2[i + 1] * x2[i + 1] +
                               x2[i + 2] * x2[i + 2] + x2[i + 3] * x2[i + 3];
                fd = fd / (sqrt(norm1 * norm2));
            }
            dtheta = ANN_MIN(acos(fd), acos(-fd)); // Quaterion angle in radians
            rho += ANN_POW(scale[i] * dtheta); // squared scaled angle
            i = i + 3;
        }
    }
    // cout << "dist = " << sqrt(rho) << endl <<endl;

    return sqrt(rho);
}

// read a point from a file stream 
// in: input file stream
// p: the point
// dim: the dimension of the product space
// returns false on EOF
ANNbool readPt(istream &in, int *p, int dim)  
{
    for (int i = 0; i < dim; i++) {
        if (!(in >> p[i])) return ANNfalse;
    }
    return ANNtrue;
}

// read a point from a file stream 
// in: input file stream
// p: the point
// dim: the dimension of the product space
// returns false on EOF
void printPt(ostream &out, ANNpoint p, int dim)  // print point
{
    out << "(" << p[0];
    for (int i = 1; i < dim; i++) {
        out << ", " << p[i];
    }
    out << ")\n";
}

int main(int argc, char **argv) {
    int MaxNeighbors = 16;  // number of nearest neighbors
    int dim = 2;            // dimension
    int m_pts = 20000;      // maximum number of data points
    int numIt = 100;        // maximum number of nearest neighbor calls

    istream *dim_in = NULL;       // input for dimension
    istream *topology_in = NULL;  // input for topology

    double d_ann, d_brute;
    int idx_ann, idx_brute;

    static ifstream dimStream;       // dimension file stream
    static ifstream topologyStream;  // topology file stream

    ANNpointArray data_pts;  // data points
    ANNpoint query_pt;       // query point
    int *topology;           // topology of the space
    ANNpoint scale;          // scaling of the coordinates
    ANNpoint result_pt;      // scaling of the coordinates
    MultiANN *MAG;           // search structure

    srand(time(NULL));
    dimStream.open("dim", ios::in);  // open query file
    if (!topologyStream) {
        cerr << "Cannot open dim file\n";
        exit(1);
    }
    dim_in = &dimStream;  // make this query stream
    (*dim_in) >> dim;     // read dimension

    query_pt = annAllocPt(dim);          // allocate query point
    data_pts = annAllocPts(m_pts, dim);  // allocate data points
    topology = new int[dim];             // allocate topology array
    scale = new double[dim];             // allocate scaling array

    topologyStream.open("topology", ios::in);  // open query file
    if (!topologyStream) {
        cerr << "Cannot open topology file\n";
        exit(1);
    }
    topology_in = &topologyStream;   // make this query stream
    readPt(*topology_in, topology, dim);  // read topology

    for (int i = 0; i < dim; i++) {
        if (topology[i] == 1)
            scale[i] = 1.0;
        else if (topology[i] == 2)
            scale[i] = 50 / PI;
        else if (topology[i] == 3)
            scale[i] = 50.0;
    }
    cout << "Topology and Scale: \n";  // print topology
    for (int i = 0; i < dim; i++) {
        cout << " topology[" << i << "] = " << topology[i] << "\t";
        cout << " scale[" << i << "] = " << scale[i] << endl;
    }
    cout << "\n";

    randomDist(dim, topology, data_pts, m_pts);       // generate random points
    randomDistPoint(dim, topology, query_pt);  // generate query point

    cout << "MaxNeighbors=" << MaxNeighbors << "\t";
    cout << "m_pts=" << m_pts << "\t";
    cout << "dim=" << dim << "\t";
    cout << "numIt=" << numIt << "\n";

    for (int ii = 0; ii < 1000; ii++) {
        cout << "\n\t test " << ii << ":" << endl;

        randomDist(dim, topology, data_pts, m_pts);       // generate random points
        randomDistPoint(dim, topology, query_pt);  // generate query point

        //***********************Construction phase**********************/

        clock_t tv1, tv2;
        double time;
        tv1 = clock();

        MAG = new MultiANN(dim, MaxNeighbors, topology,
                           scale);  // create a data structure
        for (int j = 0; j < m_pts; j++) {
            MAG->AddPoint(data_pts[j], data_pts[j]);  // add new data point
        }

        // ANN *a = new ANN(data_pts, 0, m_pts-1, m_pts, dim, topology, scale,
        // 1);

        tv2 = clock();
        time = (tv2 - tv1) / (CLOCKS_PER_SEC / (double)1000.0);
        cout << "Construction MPNN time:" << time / numIt << "sec\n";
        tv1 = clock();

        //*************************Query phase*******************************/

        double d;
        for (int j = 0; j < numIt; j++) {
            // randomDistPoint(dim, topology, query_pt);				// generate query
            // point
            d_ann = INFINITY;
            result_pt = (ANNpoint)MAG->NearestNeighbor(
                query_pt, idx_ann,
                d_ann);  // compute nearest neighbor using library
            // idx_ann = a->NearestNeighbor(topology, scale, query_pt, d_ann);
            // // compute nearest neighbor using library
        }

        tv2 = clock();
        time = (tv2 - tv1) / (CLOCKS_PER_SEC / (double)1000.0);
        cout << "MPNN query time:" << time / numIt << "sec\n\n";
        tv1 = clock();

        for (int j = 0; j < numIt; j++) {
            // randomDistPoint(dim, topology, query_pt);				// generate query
            // point
            d_brute = INFINITY;  // compare the obtained result with the brute
                                 // force result
            for (int i = 0; i < m_pts; i++) {
                d = Metric(query_pt, data_pts[i], dim, topology, scale);
                if (d_brute > d) {
                    d_brute = d;
                    idx_brute = i;
                }
            }
        }

        tv2 = clock();
        time = (tv2 - tv1) / (CLOCKS_PER_SEC / (double)1000.0);
        cout << "brute force time:" << time / numIt << "sec\n\n";
        tv1 = clock();


        if (MAG) {
            MAG->ResetMultiANN();
            MAG = NULL;
        }

        cout << "ANN distance = " << d_ann << endl;
        cout << "nearest neighbor";
        printPt(cout, result_pt, dim);

        cout << "Brute distance = " << d_brute << endl;
        cout << "nearest neighbor";
        printPt(cout, data_pts[idx_brute], dim);
    }
}

