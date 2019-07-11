#ifndef __MANIFOLD_H__
#define __MANIFOLD_H__
#include<deal.II/grid/manifold.h>
//using namespace dealii; DON'T use "using namespace" in header file!

//----------------------------------------------------------------------------------------------------
//This is a curve boundary manifold object used in ringleb's flow. 
//it's used on the left and right side of the domain, if necessary, can also be used on interior cells
//----------------------------------------------------------------------------------------------------
class RinglebSide : public dealii::ChartManifold<2,2>
{
public:
   virtual dealii::Point<2> pull_back(const dealii::Point<2>& /*space_point*/x) const override;
   virtual dealii::Point<2> push_forward(const dealii::Point<2>& /*chart_point*/psi_theta) const override;
};

//----------------------------------------------------------------------------------------------------
// Another Manifold
// this is used at the top boundary of rinleb's flow domain(local coordinate now is psi_V, not psi_theta)
//----------------------------------------------------------------------------------------------------
class RinglebTop : public dealii::ChartManifold<2,2>
{
public:
   virtual dealii::Point<2> pull_back(const dealii::Point<2>& /*space_point*/x) const override;
   virtual dealii::Point<2> push_forward(const dealii::Point<2>& /*chart_point*/psi_V) const override;
};

//----------------------------------------------------------------------------------------------------
// NACA0012 airfoil Manifold. Just project the candidate point onto the true boundary in the normal
// direction to the boundary edge. see my master thesis.
//----------------------------------------------------------------------------------------------------
class NACA0012: public dealii::Manifold<2,2>
{
private: 
   //used in newton method for projection to manifold
   void func(const double& m, const dealii::Point<2>& P, const double& x, double& f, double& d) const; 
public:
   //note that candidate point is calculated as the convex combination of the given points 
   //by Manifold::get_intermediate_point()
   virtual dealii::Point<2> project_to_manifold (const std::vector<dealii::Point<2>>& surrounding_points,
                                                 const dealii::Point<2>&              candidate) const;
};

//given x-coordinate of the point at the boundary, return its y-coordinates
//so this function describes the shape of the airfoil 
template <typename T>
T naca0012(const T& x)
{
   T x2 = x * x;
   T x3 = x * x2;
   T x4 = x * x3;
   T y = 0.594689181*(0.298222773*sqrt(x) - 0.127125232*x - 0.357907906*x2 
                      + 0.291984971*x3 - 0.105174606*x4);
   return y;
}

struct NACA      //serve as a namespace. use it as: NACA::set_curved_boundaries(triangulation);
{
   static void set_curved_boundaries(dealii::Triangulation<2>& triangulation, int boundary_id)
   {
   //we move points on airfoil to actual airfoil curve since we are not sure Gmsh has ensured this
   dealii::Triangulation<2>::active_cell_iterator cell = triangulation.begin_active(),
                                                  endc = triangulation.end();
   for(; cell!=endc; ++cell)
      for(unsigned int f=0; f<dealii::GeometryInfo<2>::faces_per_cell; ++f)
      {
         if(cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == boundary_id)
         {
            cell->face(f)->set_manifold_id(cell->face(f)->boundary_id());
            for(unsigned int v=0; v<dealii::GeometryInfo<2>::vertices_per_face; ++v)
            {
               dealii::Point<2>& space_point = cell->face(f)->vertex(v);
               double x = space_point[0];
               //if x is between 0 and 1, it is what it is, otherwise force it to be 0 or 1
               x = std::min(x, 1.0);
               x = std::max(x, 0.0);  
               //force space_point[1] to coincide with the given curve function
               double y = naca0012(x);
               if(space_point[1]>0)
                  space_point[1] = y;
               else 
                  space_point[1] = -y;
            }
         } 
      }
   //attach manifold description
   static NACA0012 airfoil;
   triangulation.set_manifold (boundary_id, airfoil);  //note that the manifold id is equal to boundary id
   }
};


#endif
