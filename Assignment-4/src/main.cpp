// Copyright (C) 2016 Daniele Panozzo <daniele.panozzo@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla Public License
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <igl/avg_edge_length.h>
#include <igl/barycenter.h>
#include <igl/local_basis.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/triangle_triangle_adjacency.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/sparse.h>
#include <igl/repdiag.h>
#include <igl/colon.h>
#include <igl/grad.h>
#include <igl/boundary_loop.h>
#include <igl/doublearea.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>

#include "tutorial_nrosy.h"

#include <fstream>

// Mesh
Eigen::MatrixXd V;
Eigen::MatrixXi F;

// Triangle-triangle adjacency
Eigen::MatrixXi TT;
Eigen::MatrixXi TTi;

// Constrained faces id
Eigen::VectorXi b;

// Cosntrained faces representative vector
Eigen::MatrixXd bc;

// Currently selected face
int selected;

// Degree of the N-RoSy field
int N = 1;

// Local basis
Eigen::MatrixXd B1, B2, B3;

// Result
Eigen::MatrixXd R;
Eigen::VectorXd R_scalar;
Eigen::MatrixXd uv_map;

// Texture image (grayscale)
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>
    texture_I;
void line_texture()
{
  int size = 128;             // Texture size
  int w = 7;                  // Line width
  int pos = size / 2 - w / 2; // Center the line
  texture_I.setConstant(size, size, 255);
  texture_I.block(0, pos, size, w).setZero();
  texture_I.block(pos, 0, w, size).setZero();
}
// Converts a representative vector per face in the full set of vectors that describe
// an N-RoSy field
void representative_to_nrosy(
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const Eigen::MatrixXd &R,
    const int N,
    Eigen::MatrixXd &Y)
{
  using namespace Eigen;
  using namespace std;

  Y.resize(F.rows() * N, 3);
  for (unsigned i = 0; i < F.rows(); ++i)
  {
    double x = R.row(i) * B1.row(i).transpose();
    double y = R.row(i) * B2.row(i).transpose();
    double angle = atan2(y, x);

    for (unsigned j = 0; j < N; ++j)
    {
      double anglej = angle + 2 * M_PI * double(j) / double(N);
      double xj = cos(anglej);
      double yj = sin(anglej);
      Y.row(i * N + j) = xj * B1.row(i) + yj * B2.row(i);
      Y.row(i * N + j) = Y.row(i * N + j) * R.row(i).norm();
    }
  }
}

// Plots the mesh with an N-RoSy field
// The constrained faces (b) are colored in red.
void plot_mesh_nrosy(
    igl::opengl::glfw::Viewer &viewer,
    Eigen::MatrixXd &V,
    Eigen::MatrixXi &F,
    int N,
    Eigen::MatrixXd &PD1,
    Eigen::VectorXi &b)
{
  using namespace Eigen;
  using namespace std;
  // Clear the mesh
  viewer.data().clear();
  viewer.data().set_mesh(V, F);
  viewer.data().set_texture(texture_I, texture_I, texture_I);

  // Expand the representative vectors in the full vector set and plot them as lines
  double avg = igl::avg_edge_length(V, F);
  MatrixXd Y;
  representative_to_nrosy(V, F, PD1, N, Y);

  MatrixXd B;
  igl::barycenter(V, F, B);

  MatrixXd Be(B.rows() * N, 3);
  for (unsigned i = 0; i < B.rows(); ++i)
    for (unsigned j = 0; j < N; ++j)
      Be.row(i * N + j) = B.row(i);

  viewer.data().add_edges(Be, Be + Y * (avg / 2), RowVector3d(0, 0, 1));

  // Highlight in red the constrained faces
  MatrixXd C = MatrixXd::Constant(F.rows(), 3, 1);
  for (unsigned i = 0; i < b.size(); ++i)
    C.row(b(i)) << 1, 0, 0;
  viewer.data().set_colors(C);
}

template <class T>
void dump_matrix(T &m, const char *file_name)
{
  std::ofstream output(file_name);
  if (!output.is_open())
  {
    std::cerr << file_name << " Open Failed\n";
    return;
  }
  for (int i = 0; i < m.rows(); ++i)
  {
    for (int j = 0; j < m.cols(); ++j)
    {
      output << m(i, j) << ' ';
    }
    output << std::endl;
  }
  output.close();
}

// It allows to change the degree of the field when a number is pressed
bool key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifier)
{
  using namespace Eigen;
  using namespace std;

  if (key == '1')
  {
    R = tutorial_nrosy(V, F, TT, b, bc, N);
    plot_mesh_nrosy(viewer, V, F, N, R, b);
    viewer.data().show_texture = false;
  }

  if (key == '2')
  {
    R = hard_nrosy(V, F, TT, b, bc, N);
    plot_mesh_nrosy(viewer, V, F, N, R, b);
    viewer.data().show_texture = false;
    dump_matrix(R, "vector_out.txt");
  }
  if (key == '3')
  {
    // 1. Get the area for each triangle
    Eigen::VectorXd A;
    igl::doublearea(V, F, A);
    Eigen::VectorXd A3 = A.replicate(3, 1);

    // 2. Get Matrix G
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    // 3. Solve the system
    Map<const VectorXd> R_flatten(R.data(), R.size());
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(G.transpose() * A3.asDiagonal() * G);
    assert(solver.info() == Success);
    R_scalar = solver.solve(G.transpose() * A3.asDiagonal() * R_flatten);
    assert(solver.info() == Success);

    // 4. Compute and plot the gradient of reconstructed scalar field
    MatrixXd gs = G * R_scalar;
    Map<MatrixXd> gs_mat(gs.data(), R.rows(), R.cols());
    MatrixXd gs_tmp = gs_mat;
    VectorXi b_tmp;
    plot_mesh_nrosy(viewer, V, F, N, gs_tmp, b_tmp);

    // 5. Compute Color Field
    MatrixXd C;
    igl::jet(R_scalar, true, C);
    viewer.data().set_colors(C);
    viewer.data().show_texture = false;
    dump_matrix(R_scalar, "scalar_out.txt");
  }
  if (key == '4')
  {
    // Clear the mesh
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);

    VectorXi bnd_V;
    igl::boundary_loop(F, bnd_V);

    MatrixXd bnd_plain;
    igl::map_vertices_to_circle(V, bnd_V, bnd_plain);

    igl::harmonic(V, F, bnd_V, bnd_plain, 1, uv_map);

    uv_map *= 5.0;

    VectorXd s = uv_map.col(1);

    MatrixXd C;
    igl::jet(s, true, C);
    viewer.data().set_colors(C);
    viewer.data().set_uv(uv_map);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
  }
  if (key == '5')
  {
    // Clear the mesh
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);

    // based on tutorial 502
    VectorXi bnd_V, b(2, 1);
    igl::boundary_loop(F, bnd_V);
    b(0) = bnd_V(0);
    b(1) = bnd_V(round(bnd_V.size() / 2));
    MatrixXd bc(2, 2);
    bc << 0, 0, 1, 0;

    igl::lscm(V, F, b, bc, uv_map);

    uv_map *= 5.0;
    VectorXd s = uv_map.col(1);

    MatrixXd C;
    igl::jet(s, true, C);
    viewer.data().set_colors(C);
    viewer.data().set_uv(uv_map);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
  }
  if (key == '6')
  {
    if (uv_map.cols() < 2)
    {
      return false;
    }
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    VectorXd s = uv_map.col(1);
    MatrixXd gs = G * s / 5.0;
    Map<MatrixXd> gs_mat(gs.data(), F.rows(), 3);

    MatrixXd gs_tmp = gs_mat;
    VectorXi b_tmp;
    plot_mesh_nrosy(viewer, V, F, N, gs_tmp, b_tmp);

    MatrixXd C;
    igl::jet(s, true, C);
    viewer.data().set_colors(C);
    viewer.data().set_uv(uv_map);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
  }
  if (key == '7')
  {
    if (uv_map.cols() < 2)
    {
      return false;
    }
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    // 1. replace one of the scalar field
    uv_map.col(1) = R_scalar * 5.0;

    // 2. Compute and plot the gradient of reconstructed scalar field
    MatrixXd gs = G * R_scalar;
    Map<MatrixXd> gs_mat(gs.data(), R.rows(), R.cols());
    MatrixXd gs_tmp = gs_mat;
    VectorXi b_tmp;
    plot_mesh_nrosy(viewer, V, F, N, gs_tmp, b_tmp);

    // 3. Plot the edited mapping
    MatrixXd C;
    igl::jet(uv_map.col(1), true, C);
    viewer.data().set_colors(C);
    viewer.data().set_uv(uv_map);
    viewer.data().show_lines = false;
    viewer.data().show_texture = true;
  }
  if (key == '8')
  {
    key_down(viewer, '7', 0);
    viewer.data().clear();
    viewer.data().set_mesh(V, F);

    auto get_face_dir = [](int face_id) -> double {
      VectorXd edge0 = uv_map.row(F(face_id, 0)) - uv_map.row(F(face_id, 1));
      VectorXd edge1 = uv_map.row(F(face_id, 0)) - uv_map.row(F(face_id, 2));
      return edge0(0) * edge1(1) - edge0(1) * edge1(0);
    };

    // 1. Get the face direction of the first triangle
    double base_face = get_face_dir(0);

    // 2. Initialize the color map
    MatrixXd C = MatrixXd::Constant(F.rows(), 3, 1);

    // 3. Get the flip indices
    vector<int> base_group;
    vector<int> flip_group;
    for (int i = 0; i < F.rows(); ++i)
    {
      double face = get_face_dir(i);
      if (base_face * face < 0)
      {
        flip_group.push_back(i);
      }
      else
      {
        base_group.push_back(i);
      }
    }
    if (flip_group.size() > base_group.size())
    {
      flip_group.swap(base_group);
    }

    // 4. Fill up C
    for (int i = 0; i < flip_group.size(); i++)
    {
      C.row(flip_group[i]) << 1, 0, 0;
    }

    // 5. dump out
    Map<VectorXi> tmp_v(flip_group.data(), flip_group.size());
    dump_matrix(tmp_v, "flip_out.txt");

    viewer.data().set_colors(C);
    viewer.data().show_lines = true;
    viewer.data().show_texture = false;
  }
  if (key == 'Z')
  {
    // Clear the mesh
    viewer.data().clear();
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);

    // 1. Get the area for each triangle
    Eigen::VectorXd A;
    igl::doublearea(V, F, A);
    Eigen::VectorXd A3 = A.replicate(3, 1);

    // 2. Get Matrix G
    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    // 3. Solve the system
    Map<const VectorXd> R_flatten(R.data(), R.size());
    SimplicialLDLT<SparseMatrix<double>> solver;
    solver.compute(G.transpose() * A3.asDiagonal() * G);
    assert(solver.info() == Success);
    Eigen::VectorXd s = solver.solve(G.transpose() * A3.asDiagonal() * R_flatten);
    assert(solver.info() == Success);

    // 4. Compute the gradient of reconstructed scalar field
    MatrixXd R_recon = G * s;
    Map<MatrixXd> R_recon_mat(R_recon.data(), R.rows(), R.cols());
    VectorXd error = (R - R_recon_mat).rowwise().norm();

    viewer.data().set_colors(error);
    viewer.data().show_texture = false;
  }

  if (key == '[' || key == ']')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    int i = b(selected);
    Vector3d v = bc.row(selected);

    double x = B1.row(i) * v;
    double y = B2.row(i) * v;
    double norm = sqrt(x * x + y * y);
    double angle = atan2(y, x);

    angle += key == '[' ? -M_PI / 16 : M_PI / 16;

    double xj = cos(angle) * norm;
    double yj = sin(angle) * norm;

    bc.row(selected) = xj * B1.row(i) + yj * B2.row(i);
    R = hard_nrosy(V, F, TT, b, bc, N);
    plot_mesh_nrosy(viewer, V, F, N, R, b);
  }

  if (key == 'Q' || key == 'W')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    bc.row(selected) = bc.row(selected) * (key == 'Q' ? 3. / 2. : 2. / 3.);

    R = hard_nrosy(V, F, TT, b, bc, N);
    plot_mesh_nrosy(viewer, V, F, N, R, b);
  }

  if (key == 'E')
  {
    if (selected >= b.size() || selected < 0)
      return false;

    b(selected) = b(b.rows() - 1);
    b.conservativeResize(b.size() - 1);
    bc.row(selected) = bc.row(bc.rows() - 1);
    bc.conservativeResize(b.size(), bc.cols());
    R = hard_nrosy(V, F, TT, b, bc, N);
    plot_mesh_nrosy(viewer, V, F, N, R, b);
  }

  return false;
}

bool mouse_down(igl::opengl::glfw::Viewer &viewer, int, int)
{
  int fid_ray;
  Eigen::Vector3f bary;
  // Cast a ray in the view direction starting from the mouse position
  double x = viewer.current_mouse_x;
  double y = viewer.core.viewport(3) - viewer.current_mouse_y;
  if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core.view,
                               viewer.core.proj, viewer.core.viewport, V, F, fid_ray, bary))
  {
    bool found = false;
    for (int i = 0; i < b.size(); ++i)
    {
      if (b(i) == fid_ray)
      {
        found = true;
        selected = i;
      }
    }

    if (!found)
    {
      b.conservativeResize(b.size() + 1);
      b(b.size() - 1) = fid_ray;
      bc.conservativeResize(bc.rows() + 1, bc.cols());

      bc.row(bc.rows() - 1) = B1.row(fid_ray);
      selected = bc.rows() - 1;

      R = hard_nrosy(V, F, TT, b, bc, N);
      plot_mesh_nrosy(viewer, V, F, N, R, b);
    }

    return true;
  }
  return false;
};

void load_constrain(const char *file)
{
  std::ifstream input(file);
  if (!input.is_open())
  {
    std::cerr << file << " Open Failed\n";
  }
  int face_id;
  double face_constrain[3];
  while (input >> face_id)
  {
    input >> face_constrain[0] >> face_constrain[1] >> face_constrain[2];
    b.conservativeResize(b.size() + 1);
    b(b.size() - 1) = face_id;
    bc.conservativeResize(bc.rows() + 1, 3);
    bc.row(bc.rows() - 1) << face_constrain[0], face_constrain[1], face_constrain[2];
  }
  input.close();
}

int main(int argc, char *argv[])
{
  using namespace std;
  using namespace Eigen;

  if (argc < 3)
  {
    const char *model = argc == 1 ? "../bumpy.off" : argv[1];
    // Load a mesh in OBJ format
    igl::readOFF(model, V, F);
    line_texture();
    // Triangle-triangle adjacency
    igl::triangle_triangle_adjacency(F, TT, TTi);

    // Compute the local_basis
    igl::local_basis(V, F, B1, B2, B3);

    // Simple constraints
    b.resize(2);
    b(0) = 0;
    b(1) = F.rows() - 1;
    bc.resize(2, 3);
    bc.row(0) = B1.row(0);
    bc.row(1) = B1.row(F.rows() - 1);

    selected = 0;

    igl::opengl::glfw::Viewer viewer;

    // Interpolate the field and plot
    key_down(viewer, '2', 0);

    // Plot the mesh
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);

    // Register the callbacks
    viewer.callback_key_down = &key_down;
    viewer.callback_mouse_down = &mouse_down;

    // Disable wireframe
    viewer.data().show_lines = false;

    // Launch the viewer
    viewer.launch();
  }
  else if (argc == 3)
  {
    // Load a given mesh from parameter
    // Load a mesh in OBJ format
    igl::readOFF(argv[1], V, F);
    line_texture();
    // Triangle-triangle adjacency
    igl::triangle_triangle_adjacency(F, TT, TTi);

    // Compute the local_basis
    igl::local_basis(V, F, B1, B2, B3);

    // Load Constrains
    load_constrain(argv[2]);

    selected = 0;

    igl::opengl::glfw::Viewer viewer;

    // Interpolate the field and plot
    key_down(viewer, '2', 0);

    // Plot the mesh
    viewer.data().set_mesh(V, F);
    viewer.data().set_texture(texture_I, texture_I, texture_I);

    // Register the callbacks
    viewer.callback_key_down = &key_down;
    viewer.callback_mouse_down = &mouse_down;

    // Disable wireframe
    viewer.data().show_lines = false;

    // Launch the viewer
    viewer.launch();
  }
  else
  {
    cout << "Usage: nrosy_field_bin <model> <constrains>\n";
  }
}
