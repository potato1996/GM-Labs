#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/slice_into.h>
#include <igl/rotate_by_quat.h>
#include <igl/cotmatrix.h>
#include <igl/per_vertex_normals.h>
#include <igl/invert_diag.h>
#include <igl/adjacency_list.h>
#include <igl/massmatrix.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <Eigen/Core>

#include "Lasso.h"
#include "Colors.h"

#include <vector>
#include <iostream>
#include <fstream>

//activate this for alternate UI (easier to debug)
//#define UPDATE_ONLY_ON_UP

using namespace std;

//vertex array, #V x3
Eigen::MatrixXd V(0, 3);
//face array, #F x3
Eigen::MatrixXi F(0, 3);
//Adjacency List
vector<vector<int>> adj_list;

//local referencing system
Eigen::MatrixXd V_base(0, 3);
Eigen::MatrixXd A_diff(0, 3);
Eigen::MatrixXd N_base(0, 3);
Eigen::VectorXi B_edge;

//mouse interaction
enum MouseMode
{
  SELECT,
  TRANSLATE,
  ROTATE,
  NONE
};
MouseMode mouse_mode = NONE;
MouseMode new_mouse_mode = NONE;
bool doit = false;
int down_mouse_x = -1, down_mouse_y = -1;

igl::min_quad_with_fixed_data<double> solver;

//for selecting vertices
std::unique_ptr<Lasso> lasso;
//list of currently selected vertices
Eigen::VectorXi selected_v(0, 1);

//for saving constrained vertices
//vertex-to-handle index, #V x1 (-1 if vertex is free)
Eigen::VectorXi handle_id(0, 1);
//list of all vertices belonging to handles, #HV x1
Eigen::VectorXi handle_vertices(0, 1);
//centroids of handle regions, #H x1
Eigen::MatrixXd handle_centroids(0, 3);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0, 3);
//index of handle being moved
int moving_handle = -1;
//rotation and translation for the handle being moved
Eigen::Vector3f translation(0, 0, 0);
Eigen::Vector4f rotation(0, 0, 0, 1.);

//per vertex color array, #V x3
Eigen::MatrixXd vertex_colors;

//function declarations (see below for implementation)
bool solve(igl::opengl::glfw::Viewer &viewer);
bool solve_init();
void get_new_handle_locations();
Eigen::Vector3f computeTranslation(igl::opengl::glfw::Viewer &viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
Eigen::Vector4f computeRotation(igl::opengl::glfw::Viewer &viewer, int mouse_x, int from_x, int mouse_y, int from_y, Eigen::RowVector3d pt3D);
void compute_handle_centroids();

bool callback_mouse_down(igl::opengl::glfw::Viewer &viewer, int button, int modifier);
bool callback_mouse_move(igl::opengl::glfw::Viewer &viewer, int mouse_x, int mouse_y);
bool callback_mouse_up(igl::opengl::glfw::Viewer &viewer, int button, int modifier);
bool callback_pre_draw(igl::opengl::glfw::Viewer &viewer);
bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers);
void onNewHandleID();
void applySelection();

// Dump a matrix to file, good for debugging
template <class T>
void dump_matrix(const T &m, string file_name)
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

bool solve_init()
{
  get_new_handle_locations();

  // Follow the textbook and compute the smooth back bone
  using namespace Eigen;
  SparseMatrix<double> A, L, M, Mi;
  igl::cotmatrix(V, F, L);
  igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_DEFAULT, M);
  igl::invert_diag(M, Mi);
  A = L * Mi * L;
  igl::min_quad_with_fixed_precompute(A, handle_vertices, SparseMatrix<double>(), true, solver);
  const VectorXd B = VectorXd::Zero(V.rows());
  bool res = igl::min_quad_with_fixed_solve(solver, B, handle_vertex_positions, VectorXd(), V_base);
  MatrixXd V_diff = V - V_base;
  assert(res);

  // Now we extract the local details
  A_diff.resize(V_base.rows(), 3);
  B_edge.resize(V_base.rows());
  igl::per_vertex_normals(V_base, F, N_base);
  for (int i = 0; i < V_base.rows(); i++)
  {
    // Search for the best local basis
    RowVector3d n = N_base.row(i);
    double max_cross_norm = 0;
    for (int neighbor : adj_list[i])
    {
      RowVector3d t = V_base.row(neighbor) - V_base.row(i);
      double curr_norm = n.cross(t).norm();
      if (curr_norm > max_cross_norm)
      {
        max_cross_norm = curr_norm;
        B_edge(i) = neighbor;
      }
    }

    // Compute Local Basis
    RowVector3d t = V_base.row(B_edge(i)) - V_base.row(i);
    RowVector3d t1 = n.cross(t).normalized();
    RowVector3d t2 = n.cross(t1).normalized();

    // Build local referencing system
    MatrixXd T(3, 3);
    T.row(0) = n;
    T.row(1) = t1;
    T.row(2) = t2;
    A_diff.row(i) = V_diff.row(i) * T.inverse();
  }
  return true;
}

bool solve(igl::opengl::glfw::Viewer &viewer)
{
  /**** Add your code for computing the deformation from handle_vertex_positions and handle_vertices here (replace following line) ****/
  using namespace Eigen;
  const VectorXd B = VectorXd::Zero(V.rows());
  bool res = igl::min_quad_with_fixed_solve(solver, B, handle_vertex_positions, VectorXd(), V_base);
  assert(res);

  igl::per_vertex_normals(V_base, F, N_base);
  for (int i = 0; i < V_base.rows(); i++)
  {
    // Compute Local Basis
    RowVector3d t = V_base.row(B_edge(i)) - V_base.row(i);
    RowVector3d n = N_base.row(i);
    RowVector3d t1 = n.cross(t).normalized();
    RowVector3d t2 = n.cross(t1).normalized();

    // Recover local details
    MatrixXd T(3, 3);
    T.row(0) = n;
    T.row(1) = t1;
    T.row(2) = t2;
    V.row(i) = V_base.row(i) + A_diff.row(i) * T;
  }
  return true;
};

void get_new_handle_locations()
{
  int count = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
    {
      Eigen::RowVector3f goalPosition = V.row(vi).cast<float>();
      if (handle_id[vi] == moving_handle)
      {
        if (mouse_mode == TRANSLATE)
          goalPosition += translation;
        else if (mouse_mode == ROTATE)
        {
          goalPosition -= handle_centroids.row(moving_handle).cast<float>();
          igl::rotate_by_quat(goalPosition.data(), rotation.data(), goalPosition.data());
          goalPosition += handle_centroids.row(moving_handle).cast<float>();
        }
      }
      handle_vertex_positions.row(count++) = goalPosition.cast<double>();
    }
}

int main(int argc, char *argv[])
{
  if (argc != 2)
  {
    cout << "Usage assignment5_bin mesh.off" << endl;
    exit(0);
  }

  // Read mesh
  igl::readOFF(argv[1], V, F);
  assert(V.rows() > 0);
  igl::adjacency_list(F, adj_list);

  handle_id.setConstant(V.rows(), 1, -1);

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;

  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);
  viewer.callback_key_down = callback_key_down;
  // Add content to the default menu window
  menu.callback_draw_viewer_menu = [&]() {
    // Draw parent menu content
    menu.draw_viewer_menu();

    // Add new group
    if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
    {

      // Expose an enumeration type
      if (ImGui::Combo("MouseMode", (int *)(&new_mouse_mode), "SELECT\0TRANSLATE\0ROTATE\0NONE\0\0"))
      {
        if (mouse_mode == SELECT)
        {
          solve_init();
        }
        mouse_mode = new_mouse_mode;
      }

      // Add a button
      if (ImGui::Button("ApplySelection"))
      {
        applySelection();
      }
      if (ImGui::Button("ClearConstraints"))
      {
        handle_id.setConstant(V.rows(), 1, -1);
      }
    }
  };

  viewer.callback_mouse_down = callback_mouse_down;
  viewer.callback_mouse_move = callback_mouse_move;
  viewer.callback_mouse_up = callback_mouse_up;
  viewer.callback_pre_draw = callback_pre_draw;

  viewer.data().clear();
  viewer.data().set_mesh(V, F);

  // Initialize selector
  lasso = std::unique_ptr<Lasso>(new Lasso(V, F, viewer));

  viewer.data().point_size = 10;
  viewer.core.set_rotation_type(igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL);

  viewer.launch();
}

bool callback_mouse_down(igl::opengl::glfw::Viewer &viewer, int button, int modifier)
{
  if (button == (int)igl::opengl::glfw::Viewer::MouseButton::Right)
    return false;

  down_mouse_x = viewer.current_mouse_x;
  down_mouse_y = viewer.current_mouse_y;

  if (mouse_mode == SELECT)
  {
    if (lasso->strokeAdd(viewer.current_mouse_x, viewer.current_mouse_y) >= 0)
      doit = true;
    else
      lasso->strokeReset();
  }
  else if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    int vi = lasso->pickVertex(viewer.current_mouse_x, viewer.current_mouse_y);
    if (vi >= 0 && handle_id[vi] >= 0) //if a region was found, mark it for translation/rotation
    {
      moving_handle = handle_id[vi];
      get_new_handle_locations();
      doit = true;
    }
  }
  return doit;
}

bool callback_mouse_move(igl::opengl::glfw::Viewer &viewer, int mouse_x, int mouse_y)
{
  if (!doit)
    return false;
  if (mouse_mode == SELECT)
  {
    lasso->strokeAdd(mouse_x, mouse_y);
    return true;
  }
  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
    if (mouse_mode == TRANSLATE)
    {
      translation = computeTranslation(viewer,
                                       mouse_x,
                                       down_mouse_x,
                                       mouse_y,
                                       down_mouse_y,
                                       handle_centroids.row(moving_handle));
    }
    else
    {
      rotation = computeRotation(viewer,
                                 mouse_x,
                                 down_mouse_x,
                                 mouse_y,
                                 down_mouse_y,
                                 handle_centroids.row(moving_handle));
    }
    get_new_handle_locations();
#ifndef UPDATE_ONLY_ON_UP
    solve(viewer);
    down_mouse_x = mouse_x;
    down_mouse_y = mouse_y;
#endif
    return true;
  }
  return false;
}

bool callback_mouse_up(igl::opengl::glfw::Viewer &viewer, int button, int modifier)
{
  if (!doit)
    return false;
  doit = false;
  if (mouse_mode == SELECT)
  {
    selected_v.resize(0, 1);
    lasso->strokeFinish(selected_v);
    return true;
  }

  if ((mouse_mode == TRANSLATE) || (mouse_mode == ROTATE))
  {
#ifdef UPDATE_ONLY_ON_UP
    if (moving_handle >= 0)
      solve(viewer);
#endif
    translation.setZero();
    rotation.setZero();
    rotation[3] = 1.;
    moving_handle = -1;

    compute_handle_centroids();

    return true;
  }

  return false;
};

bool callback_pre_draw(igl::opengl::glfw::Viewer &viewer)
{
  // Initialize vertex colors
  vertex_colors = Eigen::MatrixXd::Constant(V.rows(), 3, .9);

  //first, color constraints
  int num = handle_id.maxCoeff();
  if (num == 0)
    num = 1;
  for (int i = 0; i < V.rows(); ++i)
    if (handle_id[i] != -1)
    {
      int r = handle_id[i] % MAXNUMREGIONS;
      vertex_colors.row(i) << regionColors[r][0], regionColors[r][1], regionColors[r][2];
    }
  //then, color selection
  for (int i = 0; i < selected_v.size(); ++i)
    vertex_colors.row(selected_v[i]) << 131. / 255, 131. / 255, 131. / 255.;
  viewer.data().set_colors(vertex_colors);

  //clear points and lines
  viewer.data().set_points(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));
  viewer.data().set_edges(Eigen::MatrixXd::Zero(0, 3), Eigen::MatrixXi::Zero(0, 3), Eigen::MatrixXd::Zero(0, 3));

  //draw the stroke of the selection
  for (unsigned int i = 0; i < lasso->strokePoints.size(); ++i)
  {
    viewer.data().add_points(lasso->strokePoints[i], Eigen::RowVector3d(0.4, 0.4, 0.4));
    if (i > 1)
      viewer.data().add_edges(lasso->strokePoints[i - 1], lasso->strokePoints[i], Eigen::RowVector3d(0.7, 0.7, .7));
  }

  // update the vertex position all the time
  viewer.data().V.resize(V.rows(), 3);
  viewer.data().V << V;

  viewer.data().dirty |= igl::opengl::MeshGL::DIRTY_POSITION;

#ifdef UPDATE_ONLY_ON_UP
  //draw only the moving parts with a white line
  if (moving_handle >= 0)
  {
    Eigen::MatrixXd edges(3 * F.rows(), 6);
    int num_edges = 0;
    for (int fi = 0; fi < F.rows(); ++fi)
    {
      int firstPickedVertex = -1;
      for (int vi = 0; vi < 3; ++vi)
        if (handle_id[F(fi, vi)] == moving_handle)
        {
          firstPickedVertex = vi;
          break;
        }
      if (firstPickedVertex == -1)
        continue;

      Eigen::Matrix3d points;
      for (int vi = 0; vi < 3; ++vi)
      {
        int vertex_id = F(fi, vi);
        if (handle_id[vertex_id] == moving_handle)
        {
          int index = -1;
          // if face is already constrained, find index in the constraints
          (handle_vertices.array() - vertex_id).cwiseAbs().minCoeff(&index);
          points.row(vi) = handle_vertex_positions.row(index);
        }
        else
          points.row(vi) = V.row(vertex_id);
      }
      edges.row(num_edges++) << points.row(0), points.row(1);
      edges.row(num_edges++) << points.row(1), points.row(2);
      edges.row(num_edges++) << points.row(2), points.row(0);
    }
    edges.conservativeResize(num_edges, Eigen::NoChange);
    viewer.data().add_edges(edges.leftCols(3), edges.rightCols(3), Eigen::RowVector3d(0.9, 0.9, 0.9));
  }
#endif
  return false;
}

bool callback_key_down(igl::opengl::glfw::Viewer &viewer, unsigned char key, int modifiers)
{
  bool handled = false;
  if (key == 'S')
  {
    mouse_mode = SELECT;
    handled = true;
  }

  if ((key == 'T') && (modifiers == IGL_MOD_ALT))
  {
    if (mouse_mode == SELECT)
    {
      solve_init();
    }
    mouse_mode = TRANSLATE;
    handled = true;
  }

  if ((key == 'R') && (modifiers == IGL_MOD_ALT))
  {
    if (mouse_mode == SELECT)
    {
      solve_init();
    }
    mouse_mode = ROTATE;
    handled = true;
  }
  if (key == 'A')
  {
    applySelection();
    callback_key_down(viewer, '1', 0);
    handled = true;
  }

  return handled;
}

void onNewHandleID()
{
  //store handle vertices too
  int numFree = (handle_id.array() == -1).cast<int>().sum();
  int num_handle_vertices = V.rows() - numFree;
  handle_vertices.setZero(num_handle_vertices);
  handle_vertex_positions.setZero(num_handle_vertices, 3);

  int count = 0;
  for (long vi = 0; vi < V.rows(); ++vi)
    if (handle_id[vi] >= 0)
      handle_vertices[count++] = vi;

  compute_handle_centroids();
}

void applySelection()
{
  int index = handle_id.maxCoeff() + 1;
  for (int i = 0; i < selected_v.rows(); ++i)
  {
    const int selected_vertex = selected_v[i];
    if (handle_id[selected_vertex] == -1)
      handle_id[selected_vertex] = index;
  }
  selected_v.resize(0, 1);

  onNewHandleID();
}

void compute_handle_centroids()
{
  //compute centroids of handles
  int num_handles = handle_id.maxCoeff() + 1;
  handle_centroids.setZero(num_handles, 3);

  Eigen::VectorXi num;
  num.setZero(num_handles, 1);
  for (long vi = 0; vi < V.rows(); ++vi)
  {
    int r = handle_id[vi];
    if (r != -1)
    {
      handle_centroids.row(r) += V.row(vi);
      num[r]++;
    }
  }

  for (long i = 0; i < num_handles; ++i)
    handle_centroids.row(i) = handle_centroids.row(i).array() / num[i];
}

//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector3f computeTranslation(igl::opengl::glfw::Viewer &viewer,
                                   int mouse_x,
                                   int from_x,
                                   int mouse_y,
                                   int from_y,
                                   Eigen::RowVector3d pt3D)
{
  Eigen::Matrix4f modelview = viewer.core.view;
  //project the given point (typically the handle centroid) to get a screen space depth
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  float depth = proj[2];

  double x, y;
  Eigen::Vector3f pos1, pos0;

  //unproject from- and to- points
  x = mouse_x;
  y = viewer.core.viewport(3) - mouse_y;
  pos1 = igl::unproject(Eigen::Vector3f(x, y, depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);

  x = from_x;
  y = viewer.core.viewport(3) - from_y;
  pos0 = igl::unproject(Eigen::Vector3f(x, y, depth),
                        modelview,
                        viewer.core.proj,
                        viewer.core.viewport);

  //translation is the vector connecting the two
  Eigen::Vector3f translation = pos1 - pos0;
  return translation;
}

//computes translation for the vertices of the moving handle based on the mouse motion
Eigen::Vector4f computeRotation(igl::opengl::glfw::Viewer &viewer,
                                int mouse_x,
                                int from_x,
                                int mouse_y,
                                int from_y,
                                Eigen::RowVector3d pt3D)
{

  Eigen::Vector4f rotation;
  rotation.setZero();
  rotation[3] = 1.;

  Eigen::Matrix4f modelview = viewer.core.view;

  //initialize a trackball around the handle that is being rotated
  //the trackball has (approximately) width w and height h
  double w = viewer.core.viewport[2] / 8;
  double h = viewer.core.viewport[3] / 8;

  //the mouse motion has to be expressed with respect to its center of mass
  //(i.e. it should approximately fall inside the region of the trackball)

  //project the given point on the handle(centroid)
  Eigen::Vector3f proj = igl::project(pt3D.transpose().cast<float>().eval(),
                                      modelview,
                                      viewer.core.proj,
                                      viewer.core.viewport);
  proj[1] = viewer.core.viewport[3] - proj[1];

  //express the mouse points w.r.t the centroid
  from_x -= proj[0];
  mouse_x -= proj[0];
  from_y -= proj[1];
  mouse_y -= proj[1];

  //shift so that the range is from 0-w and 0-h respectively (similarly to a standard viewport)
  from_x += w / 2;
  mouse_x += w / 2;
  from_y += h / 2;
  mouse_y += h / 2;

  //get rotation from trackball
  Eigen::Vector4f drot = viewer.core.trackball_angle.coeffs();
  Eigen::Vector4f drot_conj;
  igl::quat_conjugate(drot.data(), drot_conj.data());
  igl::trackball(w, h, float(1.), rotation.data(), from_x, from_y, mouse_x, mouse_y, rotation.data());

  //account for the modelview rotation: prerotate by modelview (place model back to the original
  //unrotated frame), postrotate by inverse modelview
  Eigen::Vector4f out;
  igl::quat_mult(rotation.data(), drot.data(), out.data());
  igl::quat_mult(drot_conj.data(), out.data(), rotation.data());
  return rotation;
}
