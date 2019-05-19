#include <Eigen/Sparse>

#include <igl/AABB.h>
#include <igl/barycentric_coordinates.h>
#include <igl/boundary_loop.h>
#include <igl/cotmatrix.h>
#include <igl/in_element.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/png/readPNG.h>
#include <igl/project.h>
#include <igl/signed_angle.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/triangle/triangulate.h>
#include <igl/unproject_ray.h>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

#define DEBUG 0
// Emm.. I don't want to type that much
typedef Eigen::MatrixXd MXd;
typedef Eigen::MatrixXi MXi;
typedef Eigen::Vector3d V3d;
typedef Eigen::Vector2d V2d;
typedef Eigen::Vector3i V3i;
typedef Eigen::Vector2i V2i;
typedef Eigen::VectorXi VXi;
float SCALE = 2.0;

MXd V_back;
MXi F_back;

// The cage boundary
MXd V_cb; // #C by 2
MXd E_cb; // #C by 2

// The object boundary
MXd V_ob; // #O by 2
MXd E_ob; // #O by 6

// The mesh bounded by the cage
MXd V_cm; // #VCM by 2
MXi F_cm; // #FCM by 3

// The mesh bounded by the object
MXd V_om; // #VOM by 2
MXi F_om; // #FOM by 3

// Texture related, deal with image
MXd UV;
Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R, G, B, A;

// Harmonic Weights
MXd W_cm;          // #VCM by #C
MXd W_om_in_cage;  // #VOM_in_cage by #C
VXi Vi_om_in_cage; // #VOM_in_cage
MXd V_om_in_cage;  // #VOM_in_cage by 3

// Global State
// Mouse state
enum mouse_state_t
{
  NONE,
  DRAW_OBJECT,
  DRAW_CAGE,
  DEFORM_NONE,
  DEFORM_DOWN,
  DEFORM_DRAG,
} STATE;

enum obj_mode_t
{
  DRAW,
  IMAGE,
} MODE,
    NEW_MODE;

// PLOT mode, bit tricks
enum plot_mode_t
{
  PLOT_CAGE_MESH = 1,
  PLOT_CAGE_BOUND = (1 << 1),
  PLOT_OBJ = (1 << 2),
};

// ------------------------- Simple helper functions -----------------------------------

void clear_cage()
{
  V_cb.resize(0, 2);
  E_cb.resize(0, 4);
};

void clear_obj()
{
  V_ob.resize(0, 2);
  E_ob.resize(0, 4);
};

void clear_cage_mesh()
{
  V_cm.resize(0, 2);
  F_cm.resize(0, 3);
};

void clear_obj_mesh()
{
  V_om.resize(0, 2);
  F_om.resize(0, 3);
}

void reset_nan_background()
{
  V_back.resize(6, 3);
  V_back << 1, 1, 0,
      1, -1, 0,
      -1, 1, 0,
      -1, -1, 0,
      1, 1, -0.00001,
      -1, -1, -0.00001;
  V_back *= SCALE;
  F_back.resize(4, 3);
  F_back << 0, 4, 2,
      0, 1, 4,
      3, 2, 5,
      3, 5, 1;
}

void reset_image_obj(string img_path)
{
  V_back.resize(4, 2);
  V_back << -1, -1,
      1, -1,
      1, 1,
      -1, 1;
  V_back *= SCALE;

  F_back.resize(2, 3);
  F_back << 0, 1, 3,
      1, 2, 3;

  UV = V_back / SCALE / 2;
  UV += 0.5 * MXd::Ones(UV.rows(), UV.cols());
  igl::png::readPNG(img_path, R, G, B, A);
}

void reset_image_bound(string img_bound_path)
{
  std::ifstream input(img_bound_path);
  if (!input.is_open())
  {
    return;
  }
  vector<double> buffer;
  double temp;
  while (input >> temp)
  {
    buffer.push_back(temp);
  }
  V_ob.resize(buffer.size() / 2, 2);
  int k = 0;
  for (int i = 0; i < V_ob.rows(); ++i)
  {
    for (int j = 0; j < V_ob.cols(); ++j, ++k)
    {
      V_ob(i, j) = buffer[k];
    }
  }
  V_ob *= SCALE;
  input.close();
  E_ob.resize(V_ob.rows(), 4);
  for (int i = 0; i < V_ob.rows(); i++)
  {
    E_ob.row(i) << V_ob(i, 0), V_ob(i, 1), V_ob((i + 1) % V_ob.rows(), 0), V_ob((i + 1) % V_ob.rows(), 1);
  }
  MXi E;
  MXd H;
  int num_obj = V_ob.rows();
  E.resize(num_obj, 2);
  for (int i = 0; i < num_obj; ++i)
  {
    E.row(i) << i, (i + 1) % num_obj;
  }
  igl::triangle::triangulate(V_ob, E, H, "a0.005q", V_om, F_om);
  UV = V_om / SCALE / 2;
  UV += 0.5 * MXd::Ones(UV.rows(), UV.cols());
}

void reset_draw(igl::opengl::glfw::Viewer &viewer)
{
  clear_cage();
  clear_cage_mesh();
  clear_obj();
  clear_obj_mesh();
  reset_nan_background();
  viewer.core.align_camera_center(V_back);
  viewer.data().show_texture = false;
  viewer.data().set_colors(Eigen::RowVector3d(0, 1, 0));
}

void reset_image(igl::opengl::glfw::Viewer &viewer)
{
  clear_cage();
  clear_cage_mesh();
  clear_obj();
  clear_obj_mesh();
  reset_image_obj("../data/texture.jpg");
  reset_image_bound("../data/object_bound.txt");
  viewer.core.align_camera_center(V_back);
  viewer.data().set_colors(Eigen::RowVector3d(1, 1, 1));
}

inline bool double_eq(double _lhs, double _rhs)
{
  double epsilon = 1e-6;
  return abs(_lhs - _rhs) < epsilon;
}

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
// -------------------- More GM related helper functions ----------------------------

inline void update_obj()
{
#if DEBUG
  V_cm = W_cm * V_cb;
#else
  V_om_in_cage = W_om_in_cage * V_cb;
  igl::slice_into(V_om_in_cage, Vi_om_in_cage, 1, V_om);
#endif
}

// Judge if two points are closed enough to be see as the same point on the screen
inline bool point_close_screen(const V2d &p1, const V2d &p2, const igl::opengl::glfw::Viewer &viewer)
{
  // project both points on to the screen
  typedef Eigen::Vector3f V3f;
  V3f p1_plane(p1(0), p1(1), 0.0f), p2_plane(p2(0), p2(1), 0.0f);
  V3f p1_screen = igl::project(p1_plane, viewer.core.view, viewer.core.proj, viewer.core.viewport);
  V3f p2_screen = igl::project(p2_plane, viewer.core.view, viewer.core.proj, viewer.core.viewport);

  // check if they are close enough
  static const float epsilon = 10.0f;
  return (p1_screen - p2_screen).norm() < epsilon;
}

// judge whether a point is inside (OR ON THE BOUNDARY) of a shape
// Assuming that V forms a loop
bool inside_shape(const MXd &V, const V2d &p)
{
  int n = V.rows();
  double res = 0;

  // at the corner
  for (int i = 0; i < n; ++i)
  {
    V2d P(V.row(i));
    if (P == p)
      return true;
  }

  for (int i = 0; i < n; ++i)
  {
    V2d P1(V.row(i));
    V2d P2(V.row((i + 1) % n));
    double temp_res = igl::signed_angle(P1, P2, p);
    if (double_eq(abs(temp_res), 0.5))
    {
      return true;
    }
    res += temp_res;
  }
  return double_eq(abs(res), 1);
}

// -------------------------- bunch of UI stuff ----------------------------------------

// This remember a difference between the mouse and the selected point
int selected = -1;
V2d mouse_diff;
// Get the projection point of the mouse onto the plane
V2d get_mouse_loc(const igl::opengl::glfw::Viewer &viewer)
{
  double x = viewer.current_mouse_x;
  double y = viewer.core.viewport(3) - viewer.current_mouse_y;
  V3d s, dir;
  igl::unproject_ray(V2d(x, y), viewer.core.view, viewer.core.proj, viewer.core.viewport, s, dir);

  double depth = s(2) / dir(2);
  double plain_x = s(0) - dir(0) * depth;
  double plain_y = s(1) - dir(1) * depth;

  V2d res(plain_x, plain_y);
  return res;
}

bool redraw(igl::opengl::glfw::Viewer &viewer, int mode)
{
  viewer.data().clear();
  viewer.data().point_size = 8;
  viewer.data().line_width = 3;
  viewer.data().show_lines = false;

  // We either polot the object mesh if we have one, or we plot the background
  if (mode & PLOT_OBJ)
  {
    if (V_om.rows() > 0)
    {
      viewer.data().set_mesh(V_om, F_om);
    }
    else
    {
      viewer.data().set_mesh(V_back, F_back);
      viewer.data().set_colors(Eigen::RowVector3d(1, 1, 1));
      if (V_ob.rows() > 0)
        viewer.data().add_points(V_ob, Eigen::RowVector3d(0.8, 0.8, 0));
      if (E_ob.rows() > 0)
        viewer.data().add_edges(E_ob.block(0, 0, E_ob.rows(), 2),
                                E_ob.block(0, 2, E_ob.rows(), 2),
                                Eigen::RowVector3d(0.8, 0.8, 0));
    }
    if (MODE == IMAGE)
    {
      viewer.data().set_uv(UV);
      viewer.data().show_texture = true;
      viewer.data().set_texture(R, G, B);
      viewer.data().set_colors(Eigen::RowVector3d(1, 1, 1));
    }
    else
    {
      viewer.data().show_texture = false;
      viewer.data().set_colors(Eigen::RowVector3d(0.8, 0.8, 0));
    }
  }

  if (V_cm.rows() > 0 && (mode & PLOT_CAGE_MESH))
  {
    viewer.data().set_mesh(V_cm, F_cm);
  }

  if (mode & PLOT_CAGE_BOUND)
  {
    if (V_cb.rows() > 0)
      viewer.data().add_points(V_cb, Eigen::RowVector3d(1.0, 0, 0));
    if (E_cb.rows() > 0)
      viewer.data().add_edges(E_cb.block(0, 0, E_cb.rows(), 2),
                              E_cb.block(0, 2, E_cb.rows(), 2),
                              Eigen::RowVector3d(0.8, 0, 0));
  }

  return true;
}

bool mouse_down(igl::opengl::glfw::Viewer &viewer, int button, int modifier)
{
  auto trace_point = [&viewer](MXd &V, MXd &E) {
    // 1. get the current mouse location
    V2d mouse_loc = get_mouse_loc(viewer);

    // 2. from a loop, end drawing
    if (V.rows() >= 3 && point_close_screen(mouse_loc, V2d(V.row(0)), viewer))
    {
      V.conservativeResize(V.rows() - 1, 2);
      STATE = NONE;
      return;
    }

    // 3. Add a new point
    if (V.rows() == 0)
    {
      V.conservativeResize(1, 2);
      V.row(0) = mouse_loc;
    }
    else
    {
      V.conservativeResize(V.rows() + 1, 2);
      V.row(V.rows() - 1) = V.row(V.rows() - 2);
      E.conservativeResize(E.rows() + 1, 4);
      E.row(E.rows() - 1) << mouse_loc(0), mouse_loc(1), mouse_loc(0), mouse_loc(1);
    }
  };

  auto select_point = [&viewer](const MXd &V, const V2d &mouse_loc) {
    for (int i = 0; i < V.rows(); ++i)
    {
      if (point_close_screen(mouse_loc, V2d(V.row(i)), viewer))
      {
        return i;
      }
    }
    return -1;
  };

  switch (STATE)
  {
  case DRAW_OBJECT:
    trace_point(V_ob, E_ob);
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    break;
  case DRAW_CAGE:
    trace_point(V_cb, E_cb);
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    break;
  case DEFORM_NONE:
  {
    V2d mouse_loc = get_mouse_loc(viewer);
    selected = select_point(V_cb, mouse_loc);
    if (selected != -1)
    {
      mouse_diff = V2d(V_cb.row(selected)) - mouse_loc;
      STATE = DEFORM_DOWN;
    }
    break;
  }
  default:
    break;
  };
  return true;
}

bool mouse_up(igl::opengl::glfw::Viewer &viewer, int button, int modifier)
{
  switch (STATE)
  {
  case DEFORM_DOWN:
    STATE = DEFORM_NONE;
    selected = -1;
    break;
  case DEFORM_DRAG:
    STATE = DEFORM_NONE;
    selected = -1;
    break;
  }
  return true;
}

bool mouse_move(igl::opengl::glfw::Viewer &viewer, int mouse_x, int mouse_y)
{
  auto trace_point = [&viewer](MXd &V, MXd &E) {
    // 1. get the current mouse location
    V2d mouse_loc = get_mouse_loc(viewer);

    // 2. update the location of the last point
    if (V.rows() == 0)
      V.resize(1, 2);
    V.row(V.rows() - 1) = mouse_loc;

    // 3. from a loop, "attach the points"
    if (V.rows() >= 3 && point_close_screen(mouse_loc, V2d(V.row(0)), viewer))
    {
      V.row(V.rows() - 1) = V.row(0);
    }

    if (E.rows() > 0)
    {
      E(E.rows() - 1, 2) = V(V.rows() - 1, 0);
      E(E.rows() - 1, 3) = V(V.rows() - 1, 1);
    }
  };

  auto drag_point = [&viewer]() {
    // 1. get the current mouse location
    V2d mouse_loc = get_mouse_loc(viewer);
    V2d new_loc = mouse_loc + mouse_diff;

    // 2. update the location of the selected cage point
    V_cb.row(selected) = new_loc;
    E_cb(selected, 0) = new_loc(0);
    E_cb(selected, 1) = new_loc(1);
    E_cb((selected + V_cb.rows() - 1) % V_cb.rows(), 2) = new_loc(0);
    E_cb((selected + V_cb.rows() - 1) % V_cb.rows(), 3) = new_loc(1);

    // 3. recalculate the object mesh and the cage mesh
    update_obj();
  };

  switch (STATE)
  {
  case DRAW_OBJECT:
    trace_point(V_ob, E_ob);
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    break;
  case DRAW_CAGE:
    trace_point(V_cb, E_cb);
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    break;
  case DEFORM_DOWN:
    STATE = DEFORM_DRAG;
  case DEFORM_DRAG:
    assert(selected != -1);
    drag_point();
#if DEBUG
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_CAGE_MESH);
#else
    redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
#endif
    break;
  default:
    break;
  };
  return true;
}

// ------------------------ OK, that's the algorithm :) -----------------------------

void compute_harmonic()
{
  using namespace Eigen;

  // 1. Triangulate the cage
  int num_cage = V_cb.rows();
  MXi E(num_cage, 2);
  MXd H;
  for (int i = 0; i < num_cage; ++i)
  {
    E.row(i) << i, (i + 1) % num_cage;
  }
  igl::triangle::triangulate(V_cb, E, H, "a0.005q", V_cm, F_cm);
  const int n = V_cm.rows();

  // 2. Triangulathe the obj, if we don't have one yet
  int num_obj = V_ob.rows();
  if (V_om.rows() == 0)
  {
    assert(V_ob.rows() > 0 && V_ob.rows() == E_ob.rows());
    E.resize(num_obj, 2);
    for (int i = 0; i < num_obj; ++i)
    {
      E.row(i) << i, (i + 1) % num_obj;
    }
    igl::triangle::triangulate(V_ob, E, H, "a0.005q", V_om, F_om);
    if (MODE == IMAGE)
    {
      UV = V_om / SCALE / 2;
      UV += 0.5 * MXd::Ones(UV.rows(), UV.cols());
    }
  }

  // 3. Get the fixed points index (boundaries)
  VectorXi bnd_V;
  igl::boundary_loop(F_cm, bnd_V);
  int boundary_size = bnd_V.size();

  // 4. Now we craft the weights on the boundaries
  auto find_k = [&bnd_V, boundary_size](int k) {
    for (int i = 0; i < boundary_size; ++i)
    {
      if (V_cm.row(bnd_V(i)) == V_cb.row(k))
      {
        return i;
      }
    }
    return -1;
  };
  int bi_0 = find_k(0);
  int bi_1 = find_k(1);
  int bi_2 = find_k(2);
  int dir = ((bi_2 < bi_0 && bi_0 < bi_1) ||
             (bi_0 < bi_1 && bi_1 < bi_2) ||
             (bi_1 < bi_2 && bi_2 < bi_0))
                ? 1
                : -1;
  int boundary_index = bi_0;
  MXd bc = MXd::Zero(boundary_size, num_cage);
  for (int i = 0; i < num_cage; ++i)
  {
    V2d curr_cage = V_cb.row(i);
    V2d next_cage = V_cb.row((i + 1) % num_cage);
    double edge_len = (next_cage - curr_cage).norm();
    for (;
         V2d(V_cm.row(bnd_V(boundary_index))) != next_cage;
         boundary_index = (boundary_index + boundary_size + dir) % boundary_size)
    {
      V2d curr_bound = V_cm.row(bnd_V(boundary_index));
      double next_weight = (curr_bound - curr_cage).norm() / edge_len;
      double curr_weight = 1 - next_weight;
      bc(boundary_index, i) = curr_weight;
      bc(boundary_index, (i + 1) % num_cage) = next_weight;
    }
  }

  // 5. Compute the harmonic coordinates of the cage mesh
  W_cm.resize(n, num_cage);
  SparseMatrix<double> L, Q;
  igl::cotmatrix(V_cm, F_cm, L);
  assert(n == L.cols());
  assert(n == L.rows());
  Q = -L;
  igl::min_quad_with_fixed_data<double> solver;
  igl::min_quad_with_fixed_precompute(Q, bnd_V, SparseMatrix<double>(), true, solver);

  const VectorXd B = VectorXd::Zero(n);
  for (int w = 0; w < num_cage; w++)
  {
    const VectorXd bcw = bc.col(w);
    VectorXd Ww;
    bool res = igl::min_quad_with_fixed_solve(solver, B, bcw, VectorXd(), Ww);
    assert(res);
    W_cm.col(w) = Ww;
  }

  // 6. compute the barycentric coordinate of the object mesh
  igl::AABB<MatrixXd, 2> tree;
  tree.init(V_cm, F_cm);
  VectorXi I_om_all, I_om;
  igl::in_element(V_cm, F_cm, V_om, tree, I_om_all);
  // 6.1 Extract I_om from I_om_all and get Vi_in_cage
  vector<int> vi_in_cage_tmp;
  for (int i = 0; i < I_om_all.size(); i++)
  {
    if (I_om_all(i) >= 0)
    {
      vi_in_cage_tmp.push_back(i);
    }
  }
  Vi_om_in_cage.resize(vi_in_cage_tmp.size());
  for (int i = 0; i < vi_in_cage_tmp.size(); i++)
  {
    Vi_om_in_cage(i) = vi_in_cage_tmp[i];
  }
  // 6.2 slice everything
  igl::slice(I_om_all, Vi_om_in_cage, 1, I_om);
  igl::slice(V_om, Vi_om_in_cage, 1, V_om_in_cage);
  MXi o_vids;
  MXd tmpA, tmpB, tmpC, bry_cord;
  igl::slice(F_cm, I_om, 1, o_vids);
  igl::slice(V_cm, o_vids.col(0), 1, tmpA);
  igl::slice(V_cm, o_vids.col(1), 1, tmpB);
  igl::slice(V_cm, o_vids.col(2), 1, tmpC);
  // 6.3 get the barycentric coordinates
  igl::barycentric_coordinates(V_om_in_cage, tmpA, tmpB, tmpC, bry_cord);

  // 7. compute the harmonic coordinate of the object mesh
  W_om_in_cage.resize(V_om_in_cage.rows(), num_cage);
  for (int i = 0; i < V_om_in_cage.rows(); ++i)
  {
    W_om_in_cage.row(i) = W_cm.row(o_vids(i, 0)) * bry_cord(i, 0) +
                          W_cm.row(o_vids(i, 1)) * bry_cord(i, 1) +
                          W_cm.row(o_vids(i, 2)) * bry_cord(i, 2);
  }
}

// ----------------------- OH my main function ------------------------------------

void test()
{
  V2d P1(0, 1);
  V2d P2(0, -1);
  V2d O(0, 0);
  cout << igl::signed_angle(P2, P1, O) << endl;
  cout << igl::signed_angle(P1, P2, O) << endl;
}

int main(int argc, char *argv[])
{
  // Initialize global variables
  STATE = NONE;
  MODE = NEW_MODE = DRAW;

  // Initialize the viewer
  igl::opengl::glfw::Viewer viewer;
  viewer.callback_mouse_down = &mouse_down;
  viewer.callback_mouse_up = &mouse_up;
  viewer.callback_mouse_move = &mouse_move;

  // By default, we use the free draw mode
  reset_draw(viewer);

  // Attach a menu plugin
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  viewer.plugins.push_back(&menu);

  menu.callback_draw_viewer_menu = [&]() {
    menu.draw_viewer_menu();
    if (ImGui::Button("RESET OBJECT", ImVec2(-1, 0)))
    {
      STATE = DRAW_OBJECT;
      if (MODE == DRAW)
      {
        reset_draw(viewer);
      }
      else
      {
        assert(MODE == IMAGE);
        reset_image(viewer);
      }
      redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    }
    if (ImGui::Button("REDRAW CAGE", ImVec2(-1, 0)))
    {
      STATE = DRAW_CAGE;

      // clean up the cage and the mesh
      clear_cage();
      clear_cage_mesh();
      if ((V_om.rows() == 0) && (V_ob.rows() != E_ob.rows()))
      {
        clear_obj();
      }
      redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
    }
    if (ImGui::Button("START DEFORM", ImVec2(-1, 0)))
    {
      if (STATE == NONE && (V_om.rows() > 0 || V_ob.size() > 0) && V_cb.size() > 0)
      {
        STATE = DEFORM_NONE;
        compute_harmonic();
        update_obj();
#if DEBUG
        redraw(viewer, PLOT_CAGE_BOUND | PLOT_CAGE_MESH);
#else
        redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
#endif
      }
    }
    if (ImGui::Combo("OBJECT MODE", (int *)&NEW_MODE, "FREE DRAW\0LOAD IMAGE\0\0"))
    {
      if (MODE != NEW_MODE)
      {
        MODE = NEW_MODE;
        STATE = NONE;
        if (MODE == DRAW)
        {
          reset_draw(viewer);
        }
        else
        {
          assert(MODE == IMAGE);
          reset_image(viewer);
        }
        redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
      }
    }
    ImGui::InputScalar("OBJECT SCALING", ImGuiDataType_Float, &SCALE);
  };
  redraw(viewer, PLOT_CAGE_BOUND | PLOT_OBJ);
  viewer.launch();
}
