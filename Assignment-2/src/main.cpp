/*** std headers ***/
#include <math.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <tuple>
#include <list>

/*** libigl helper headers ***/
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/copyleft/marching_cubes.h>
#include <igl/slice.h>
#include <igl/facet_components.h>

using namespace std;
using Viewer = igl::opengl::glfw::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrained_points;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrained_values;

// Parameter: degree of the polynomial
unsigned int polyDegree = 1;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;
unsigned int mesh2radiu = 15;
unsigned int radiu2epsilon = 10;

// Parameter: grid resolution
unsigned int resolution = 40;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Data Structure to help with local search on **Original** points
vector<int> op2og;
vector<list<int>> og2op;
vector<int> odims;
vector<double> ogbase;

// Data Structure to help with local search on **Constrained** points
vector<int> cp2cg;
vector<list<int>> cg2cp;
vector<int> cdims;
vector<double> cgbase;

#define DEBUG 0

// Helper Functions
int get_bcol() {
    int count = 0;
    for (int i = 0; i <= polyDegree; i++) {
        count += (i*(i + 1) / 2 + i + 1);
    }
    return count;
}
bool double_eq(double e1, double e2){
    return fabs(e1 - e2) < 0.000001;
}

int gid2id(vector<int>& gid, vector<int>& dims){
    return gid[0] * dims[1] * dims[2] 
            + gid[1] * dims[2]
            + gid[2];
}

vector<int> id2gid(int id, vector<int>& dims){
    return {id / (dims[1] * dims[2]), 
            (id / dims[2]) % dims[1], 
            id % dims[2]};
}

vector<int> get_neighbor_blocks(int id, vector<int>& dims){
    vector<int> res;
    auto is_valid = [&dims](vector<int>& gid){
        for(int i = 0; i < 3; ++i){
            if(gid[i] < 0 || gid[i] >= dims[i]){
                return false;
            }
        }
        return true;
    };
    for(int i = -1; i <= 1; ++i){
        for(int j = -1; j <= 1; ++j){
            for(int k = -1; k <= 1; ++k){
                auto gid = id2gid(id, dims);
                gid[0] += i;
                gid[1] += j;
                gid[2] += k;
                if(is_valid(gid)){
                    res.push_back(gid2id(gid, dims));
                }
            }
        }
    }
    return res;
}

double wendland(double r){
    if(abs(r) < wendlandRadius){
        double r2h = r/wendlandRadius;
        return pow(1 - r2h, 4.0) * (4 * r2h + 1);
    }
    else{
        cout << "We should never reach here\n";
        return 0.0;
    }
}

void recompute_wendland(){
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();
    wendlandRadius = (bb_max - bb_min).norm() / double(mesh2radiu);
}

void build_local_search(Eigen::MatrixXd& points,
        vector<int>& p2g,
        vector<list<int>>& g2p,
        vector<int>& dims,
        vector<double>& gbase){
    int num_points = points.rows();

    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = points.colwise().minCoeff();
    bb_max = points.colwise().maxCoeff();

    gbase.resize(3);
    for(int i = 0; i < 3; ++i){
        gbase[i] = bb_min(i) - wendlandRadius;
    }

    // Re-calculate our grid dimensions;
    dims.resize(3);
    Eigen::Vector3d max_resolution = (bb_max - bb_min)/wendlandRadius;
    for(int i = 0; i < 3; ++i){
        dims[i] = int(max_resolution(i)) + 3;
    }

    // Set up our mappings
    p2g.clear();
    g2p.clear();
    p2g.resize(num_points);
    g2p.resize(dims[0] * dims[1] * dims[2]);
    for(int i = 0; i < num_points; ++i){
        vector<int> index(3);
        for(int j = 0; j < 3; ++j){
            index[j] = int((points(i, j) - gbase[j])/wendlandRadius);
        }
        int grid_id = gid2id(index, dims);
        p2g[i] = grid_id;
        g2p[grid_id].push_back(i);
    }

    cout << "Finish Point-Grid Mapping\n";
}


// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callback_key_down(Viewer& viewer, unsigned char key, int modifiers);

// Creates a grid_points array
void createGrid() {
    recompute_wendland();

    // Grid bounds: non-axis-aligned bounding box (2.3)

    // 1. build the cov matrix
    Eigen::MatrixXd centered = P.rowwise() - P.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(P.rows());
    
    // 2. PCA solver
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
    Eigen::MatrixXd evecs = solver.eigenvectors();
    
    // 3. Get the new axis
    Eigen::MatrixXd axis = evecs.colwise().normalized();
    if ((Eigen::Vector3d(axis.col(0)).cross(Eigen::Vector3d(axis.col(1)))).dot(Eigen::Vector3d(axis.col(2))) < 0){
        axis.col(2) = -axis.col(2);
    }

    // 4. Find the min/max under the new axis
    Eigen::MatrixXd new_points = P * axis;
    Eigen::RowVector3d bb_min = (new_points.colwise().minCoeff()).transpose();
    Eigen::RowVector3d bb_max = (new_points.colwise().maxCoeff()).transpose();

    // 5. Bounding box dimensions
    for(int i = 0; i < 3; ++i){
        bb_min(i) -= 2.0 * wendlandRadius / radiu2epsilon;
        bb_max(i) += 2.0 * wendlandRadius / radiu2epsilon;
    }
    Eigen::RowVector3d dim = bb_max - bb_min;

    // 6. Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);

    // 7. 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);
    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = bb_min + Eigen::RowVector3d(x * dx, y * dy, z * dz);
            }
        }
    }

    // 8. transpose the grid back
    grid_points = grid_points * (axis.inverse());

    cout << "Finish Creating Grid\n";
}

void evaluateImplicitFunc() {
    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    recompute_wendland();

    // Calculate the mapping with the constrained points
    build_local_search(constrained_points,
        cp2cg,
        cg2cp,
        cdims,
        cgbase);

    auto construct_b = [](Eigen::RowVector3d& point){
        int B_col = get_bcol();
        Eigen::RowVectorXd b(B_col);
        for(int k = 0, col_id = 0; k <= polyDegree; k++){
            for(int x_p = 0; x_p <= k; x_p++){
                for(int y_p = 0; x_p + y_p <= k; y_p++, col_id++){
                    int z_p = k - x_p - y_p;
                    b(col_id) = pow(point(0), x_p) * pow(point(1), y_p) * pow(point(2), z_p);
                }
            }
        }
        return b;
    };

    // Evaluate signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                Eigen::RowVector3d grid_pos = grid_points.row(index);

                // 1. Which grid does this point lies in?
                vector<int> grid_index(3);
                for(int i = 0; i < 3; ++i){
                    grid_index[i] = floor((grid_pos(i) - cgbase[i]) / wendlandRadius);
                }

                // Validate that this point lies in our pre-computed locating grid
                auto validate_index = [](vector<int>& id){
                    for(int i = 0; i < 3; ++i){
                        if(id[i] < 0 || id[i] >= cdims[i]){
                            return false;
                        }
                    }
                    return true;
                };
                if(!validate_index(grid_index)){
                    // There's no point with in h to this point
                    grid_values[index] = 1000.0;
                    continue;
                }

                // 2. Extract interested points (i.e. distance smaller than wendland radiu)
                auto neighbor_blocks = get_neighbor_blocks(gid2id(grid_index, cdims), cdims);
                vector<int> radiu_points_v;
                vector<double> radiu_points_dis;
                for(int block: neighbor_blocks){
                    for(int point: cg2cp[block]){
                        double dis = (Eigen::RowVector3d(constrained_points.row(point)) - grid_pos).norm();
                        if(dis < wendlandRadius){
                            radiu_points_v.push_back(point);
                            radiu_points_dis.push_back(dis);
                        }
                    }
                }

                int num_radiu_points = radiu_points_v.size();

                if(num_radiu_points < get_bcol()){
                    // There's no enough points with in h to this point
                    grid_values[index] = 1000.0;
                    continue;
                }

                // 3. Construct B and d
                Eigen::VectorXd d(num_radiu_points);

                Eigen::MatrixXd B;
                int B_row = num_radiu_points;
                int B_col = get_bcol();
                B.resize(B_row, B_col);

                for(int i = 0; i < num_radiu_points; ++i){
                    Eigen::RowVector3d point = constrained_points.row(radiu_points_v[i]);
                    B.row(i) = construct_b(point);
                    d(i) = constrained_values(radiu_points_v[i]);
                }

                // 4. Construct W
                Eigen::VectorXd ws(num_radiu_points);
                for(int i = 0; i < num_radiu_points; i++){
                    ws(i) = wendland(radiu_points_dis[i]);
                }
                auto W = ws.asDiagonal();

                // 5. Solve the equation
                Eigen::MatrixXd lhs = B.transpose() * W * B;
                Eigen::VectorXd rhs = B.transpose() * W * d;
                Eigen::VectorXd a = lhs.inverse() * rhs;

                // 6. Compute the value at the grid point
                grid_values[index] = construct_b(grid_pos) * a;

                if(DEBUG && double_eq(grid_values[index], 0)){
                    // print out our debug informations..
                    cout << num_radiu_points << endl
                        << W.diagonal() << endl
                        << B << endl 
                        << lhs << endl
                        << rhs << endl
                        << a << endl
                        << construct_b(grid_pos) << endl;
                }
            }
        }
    }
}

void evaluateImplicitFuncOpt2() {
    // Scalar values of the grid points (the implicit function values)
    grid_values.resize(resolution * resolution * resolution);

    recompute_wendland();

    // Calculate the mapping with the original points
    build_local_search(P, op2og, og2op, odims, ogbase);


    auto construct_b = [](Eigen::RowVector3d& point){
        int B_col = get_bcol();
        Eigen::RowVectorXd b(B_col);
        for(int k = 0, col_id = 0; k <= polyDegree; k++){
            for(int x_p = 0; x_p <= k; x_p++){
                for(int y_p = 0; x_p + y_p <= k; y_p++, col_id++){
                    int z_p = k - x_p - y_p;
                    b(col_id) = pow(point(0), x_p) * pow(point(1), y_p) * pow(point(2), z_p);
                }
            }
        }
        return b;
    };

    // Evaluate signed distance function at each gridpoint.
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                
                // Linear index of the point at (x,y,z)
                int index = x + resolution * (y + resolution * z);
                Eigen::RowVector3d grid_pos = grid_points.row(index);

                // 1. Which grid does this point lies in?
                vector<int> grid_index(3);
                for(int i = 0; i < 3; ++i){
                    grid_index[i] = floor((grid_pos(i) - ogbase[i]) / wendlandRadius);
                }

                // Validate that this point lies in our pre-computed locating grid
                auto validate_index = [](vector<int>& id){
                    for(int i = 0; i < 3; ++i){
                        if(id[i] < 0 || id[i] >= odims[i]){
                            return false;
                        }
                    }
                    return true;
                };
                if(!validate_index(grid_index)){
                    // There's no point with in h to this point
                    grid_values[index] = 1000.0;
                    continue;
                }

                // 2. Extract interested points (i.e. distance smaller than wendland radiu)
                auto neighbor_blocks = get_neighbor_blocks(gid2id(grid_index, odims), odims);
                vector<int> radiu_points_v;
                vector<double> radiu_points_dis;
                for(int block: neighbor_blocks){
                    for(int point: og2op[block]){
                        double dis = (Eigen::RowVector3d(P.row(point)) - grid_pos).norm();
                        if(dis < wendlandRadius){
                            radiu_points_v.push_back(point);
                            radiu_points_dis.push_back(dis);
                        }
                    }
                }

                int num_radiu_points = radiu_points_v.size();

                if(num_radiu_points < get_bcol()){
                    // There's no enough points with in h to this point
                    grid_values[index] = 1000.0;
                    continue;
                }

                // 3. Construct B and d
                Eigen::VectorXd d(num_radiu_points);

                Eigen::MatrixXd B;
                int B_row = num_radiu_points;
                int B_col = get_bcol();
                B.resize(B_row, B_col);

                for(int i = 0; i < num_radiu_points; ++i){
                    Eigen::RowVector3d point = P.row(radiu_points_v[i]);
                    B.row(i) = construct_b(point);
                    d(i) = (grid_pos - point).dot(Eigen::RowVector3d(N.row(radiu_points_v[i])));
                }

                // 4. Construct W
                Eigen::VectorXd ws(num_radiu_points);
                for(int i = 0; i < num_radiu_points; i++){
                    ws(i) = wendland(radiu_points_dis[i]);
                }
                auto W = ws.asDiagonal();

                // 5. Solve the equation
                Eigen::MatrixXd lhs = B.transpose() * W * B;
                Eigen::VectorXd rhs = B.transpose() * W * d;
                Eigen::VectorXd a = lhs.inverse() * rhs;

                // 6. Compute the value at the grid point
                grid_values[index] = construct_b(grid_pos) * a;

                if(DEBUG && double_eq(grid_values[index], 0)){
                    // print out our debug informations..
                    cout << num_radiu_points << endl
                        << W.diagonal() << endl
                        << B << endl 
                        << lhs << endl
                        << rhs << endl
                        << a << endl
                        << construct_b(grid_pos) << endl;
                }
            }
        }
    }
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}

bool callback_key_down(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        viewer.data().clear();
        viewer.core.align_camera_center(P);
        viewer.data().point_size = 7;
        viewer.data().add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data().clear();
        viewer.core.align_camera_center(P);

        // normalize the normals
        N.rowwise().normalize();

        // re-compute wendland radius
        recompute_wendland();

        // re-build the mapping
        build_local_search(P, op2og, og2op, odims, ogbase);

        int num_points = P.rows();

        // Build up constrain points
        constrained_points.resize(3 * num_points, 3);
        constrained_values.resize(3 * num_points);
        
        Eigen::MatrixXd constrained_colors;
        constrained_colors.resize(3 * num_points, 3);

        int changed_count = 0;

        // This helper function checks whether target is nearest point to the candidate
        auto check_mindis = [](const Eigen::Vector3d& candidate, const int target){
            double mindis = (candidate - Eigen::Vector3d(P.row(target))).norm();
            
            // which grid candidate lies in?
            vector<int> grid_index(3);
            for(int i = 0; i < 3; ++i){
                grid_index[i] = int((candidate(i) - ogbase[i]) / wendlandRadius);
            }

            // Go through all the candidate points
            auto neighbor_blocks = get_neighbor_blocks(gid2id(grid_index, odims), odims);
            for(int block: neighbor_blocks){
                for(int point: og2op[block]){
                    if(P.row(point) == P.row(target))continue;
                    double dis = (Eigen::Vector3d(P.row(point)) - candidate).norm();
                    if(dis < mindis) return false;
                }
            }
            return true;
        };

        // Add the original points
        for(int i = 0; i < num_points; ++i){
            constrained_points.row(i) = P.row(i);
            constrained_values(i) = 0;
            constrained_colors.row(i) = Eigen::RowVector3d(1, 0, 0);
        }

        cout << "Original Point Added\n";

        // This make sure that we are safe to only check the surranding 3x3x3 grids
        double global_epsilon = wendlandRadius / double(radiu2epsilon);

        // Add the positive normal points
        for(int i = 0; i < num_points; ++i){
            double epsilon = global_epsilon;
            while(true){
                Eigen::Vector3d pos_norm = P.row(i) + N.row(i) * epsilon;
                constrained_points.row(i + num_points) = pos_norm;
                constrained_values(i + num_points) = epsilon;
                constrained_colors.row(i + num_points) = Eigen::RowVector3d(0, 1, 0);

                // Check that we satisfy the constrain
                if(!check_mindis(constrained_points.row(i + num_points), i)){
                    epsilon *= 0.5;
                }
                else{
                    break;
                }
            }
            if(epsilon != global_epsilon) changed_count++;
        }

        cout << "Pos-Norm Added\n";
        
        // Add the negative normal points
        for(int i = 0; i < num_points; ++i){
            double epsilon = global_epsilon;
            while(true){
                Eigen::Vector3d neg_norm = P.row(i) - N.row(i) * epsilon;
                constrained_points.row(i + 2 * num_points) = neg_norm;
                constrained_values(i + 2 * num_points) = -epsilon;
                constrained_colors.row(i + 2 * num_points) = Eigen::RowVector3d(0, 0, 1);

                // Check that we satisfy the constrain
                if(!check_mindis(constrained_points.row(i + 2 * num_points), i)){
                    epsilon *= 0.5;
                }
                else{
                    break;
                }
            }
            if(epsilon != global_epsilon) changed_count++;
        }
        
        cout << "Neg-Norm Added\n";

        cout << "Total Constrain Points: " << constrained_points.rows() << endl
            << "Constrained Points with Adjusted Epsilon: " << changed_count << endl;

        viewer.data().point_size = 5;
        viewer.data().add_points(constrained_points, constrained_colors);
    }

    if (key == '3') {
        if(constrained_points.rows() == 0){
            cout << "Error: constrain points not computed, press key \"2\" first";
            return true;
        }
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);

        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data().clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        // Remove small components
        Eigen::VectorXi cids(F.rows());
        igl::facet_components(F, cids);
        unordered_map<int, vector<int>> c2f;
        int max_cid = -1;
        int max_size = 0;
        for(int i = 0; i < F.rows(); ++i){
            c2f[cids(i)].push_back(i);
            if(c2f[cids(i)].size() > max_size){
                max_cid = cids(i);
                max_size = c2f[cids(i)].size();
            }
        }

        Eigen::MatrixXi filtered_F(max_size, 3);
        for(int i = 0; i < max_size; ++i){
            filtered_F.row(i) = F.row(c2f[max_cid][i]);
        }
        F = filtered_F;

        igl::writeOFF("out.off", V, F);

        igl::per_face_normals(V, F, FN);
        viewer.data().set_mesh(V, F);
        viewer.data().show_lines = true;
        viewer.data().show_faces = true;
        viewer.data().set_normals(FN);
    }

    if (key == '5') {
        // Show grid points with colored nodes and connected with lines
        viewer.data().clear();
        viewer.core.align_camera_center(P);

        /*** begin: sphere example, replace (at least partially) with your code ***/
        // Make grid
        createGrid();

        // Evaluate implicit function (Using the method in Optional Task 2)
        evaluateImplicitFuncOpt2();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            }
            else {
                if (value > 0)
                    grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.data().point_size = 8;
        viewer.data().add_points(grid_points, grid_colors);
        viewer.data().add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ex2_bin mesh.off" << endl;
        exit(0);
    }

    // Read points and normals
    igl::readOFF(argv[1],P,F,N);

    if(N.rows() == 0){
        cout << "ERROR: Normals not loaded\n";
        exit(1);
    }
    else{
        cout << N.rows() << endl;
    }

    Viewer viewer;
    viewer.callback_key_down = callback_key_down;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    viewer.plugins.push_back(&menu);

    menu.callback_draw_viewer_menu = [&]() {
        menu.draw_viewer_menu();
        // Add widgets to the sidebar.
        if (ImGui::CollapsingHeader("Reconstruction Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputScalar("resolution", ImGuiDataType_U32, &resolution);
            ImGui::Combo("polyDegree", (int*)&polyDegree, "0\0 1\0 2\0\0");
            if (ImGui::Button("Reset Grid", ImVec2(-1,0))) {
                // Switch view to show the grid
                callback_key_down(viewer, '3', 0);
            }
        }
        if (ImGui::CollapsingHeader("Constrain Options", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputScalar("mesh/radiu", ImGuiDataType_U32, &mesh2radiu);
            ImGui::InputScalar("radiu/epsilon", ImGuiDataType_U32, &radiu2epsilon);
            if (ImGui::Button("Reset Constrains", ImVec2(-1,0))) {
                // Switch view to show the grid
                callback_key_down(viewer, '2', 0);
            }
        }
    };

    viewer.launch();
}
