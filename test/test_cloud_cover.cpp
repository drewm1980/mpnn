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
// topologies: individual topologies in the cross product: 
// 2 for -PI to PI
// 3 for quaternion. MUST APPEAR 4 times consecutively!
// 1 for -100 to 100
// p: the point in the product space
//
// Doesn't do any memory allocation; p must be large enough!
void randomDistPoint(int dim, const int *topologies, ANNpoint &p) {
    for (int kk = 0; kk < dim; kk++) {
        double tmp = rand() / (2.14783e+09);
        if (topologies[kk] == 2) {
            p[kk] = (2 * PI * tmp - PI);  // region -PI PI
        } else if (topologies[kk] == 3) {
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
            rho += ANN_POW(scale[i] * dtheta); // squared scaled angle in radians
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

struct Quaternion
{
  double p[4];
};
vector<Quaternion> load_quaternion_cloud(const string filename)
{
    ifstream stream(filename); 
    vector<Quaternion> cloud;
    Quaternion q;
    int lines_loaded = 0;
    while (!stream.eof()) {
        for (int i = 0; i < 4; i++) stream >> q.p[i];
        cloud.push_back(q);
        lines_loaded += 1;
    }
    cloud.pop_back();
    cout << "Loaded " << cloud.size() << " points!" << endl;
    return cloud;
}

int main(int argc, char **argv) {
    string filename;
    switch (argc) {
        case 1:
            cout << "Loading the quaternion file from disk..." << endl;
            filename = string("simple_0.qua");
            break;
        case 2:
            filename = string(argv[1]);
            break;
        default:
            cout << "Wrong number of arguments!" << endl;
            exit(EXIT_FAILURE);
    }
    auto cloud = load_quaternion_cloud(filename);

    int max_points = cloud.size();      // maximum number of data points
    int dimension = 4;            
    ANNpoint first_point = &(cloud[0].p[0]); 
    vector<double*> pointers_to_points;
    // ANN requires an entire vector of pointers even if the array is dense!
    for (int i = 0; i < max_points; i++) {
        pointers_to_points.push_back(first_point + 4*i);
    }
    ANNpointArray data_points = &pointers_to_points[0]; 

    cout << "Creating the search data structure..." << endl;
    int topology[] = {3, 3, 3, 3}; // Code for a single quaternion
    double _scale = 1.0;
    double scale[] = {_scale, _scale, _scale, _scale};
    int MaxNeighbors = 16;  // number of nearest neighbors
    MultiANN MAG(dimension, MaxNeighbors, topology, scale);
    for (int j = 0; j < max_points; j++) {
      MAG.AddPoint(data_points[j], data_points[j]);  
    }

    cout << "Generating Random Query Point" << endl;
    ANNpoint query_pt = annAllocPt(dimension);          // allocate query point
    //randomDistPoint(dimension, topology, query_pt);
    query_pt[0] = 0.0;
    query_pt[1] = 0.0;
    query_pt[2] = 0.0;
    query_pt[3] = 1.0;

    {
        cout << "Calling single point nearest neighbor..." << endl;
        double d_ann = INFINITY;
        int idx_ann;
        auto result_pt = (ANNpoint)MAG.NearestNeighbor(
            query_pt, idx_ann, d_ann);  // single nearest neighbor
        printPt(cout << "query_pt = ", query_pt, dimension);
        printPt(cout << "result_pt = ", result_pt, dimension);
        cout << "ANN distance = " << d_ann << " rad" << endl;
        double d_degrees = d_ann * 180.0 / PI;
        cout << "ANN distance = " << d_degrees << " degrees" << endl;
    }
    {
        cout << "Calling multiple point nearest neighbor..." << endl;
        double d_ann = INFINITY;

        double best_dist[MaxNeighbors];
        ANNpoint p_best_dist[MaxNeighbors];
        for(int i=0; i<MaxNeighbors; i++) p_best_dist[i] = best_dist+i;

        int best_neighbor_indeces[MaxNeighbors];
        int* p_best_neighbor_indeces[MaxNeighbors];
        for(int i=0; i<MaxNeighbors; i++) p_best_neighbor_indeces[i] = best_neighbor_indeces+i;
        void *best_ptr[MaxNeighbors];
        void **p_best_ptr[MaxNeighbors];
        for(int i=0; i<MaxNeighbors; i++) p_best_ptr[i] = best_ptr+i;
        MAG.NearestNeighbor(query_pt, p_best_dist[0], p_best_neighbor_indeces[0],
                            p_best_ptr[0]);  // multiple nearest neighbor

        cout << "Nearest Neighbors:" << endl;
        for(int i=0; i<MaxNeighbors; i++)
        {
          int best_neighbor_index = best_neighbor_indeces[i];
          //cout << "best_neighbor_index = " << best_neighbor_index << endl;
          cout << best_dist[i]*180.0/PI << " deg: ";
          printPt(cout, pointers_to_points[best_neighbor_index], dimension);
        }

    }
}

