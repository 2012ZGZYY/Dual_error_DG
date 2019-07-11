//#include<deal.II/grid/tria_accessor.h>
//#include<deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_accessor.h>
#include <Sacado.hpp>
#include "dual_functional.h"
#include "equation.h"
#define Pi 3.1415926
using namespace dealii;

namespace DualFunctional
{
   template<int dim>
   PointValueEvaluation<dim>::
   PointValueEvaluation (const Point<dim>& evaluation_point)
                        :
                        evaluation_point (evaluation_point)
   {}

   template<int dim>
   void 
   PointValueEvaluation<dim>::
   assemble_rhs(const DoFHandler<dim>& dof_handler,
                const dealii::Mapping<dim>&   mapping,
                const Vector<double>&  primal_solution,
                Vector<double>&        rhs_dual) const
   { 
   /*if use linear fe_system, then there are 16 local dofs for a cell. to evaluate the density value 
    of a point belongs to this cell, we have the following:
    J(u) = rho(x0) = J(U) = U^e_8*phi_0(x0) + U^e_9*phi_1(x0) + U^e_10*phi_2(x0) + U^e_10*phi_3(x0)
    U^e_8~10 are local dof_values correspond to quantity rho. phi_0~3 are local shape functions
    so the above equation denotes the density at point x0.
    so under the local perspective, J'[u](du) = J'(U)DU = d(rho(x0)), 
    then J'(U) = (0, 0, 0, 0,| 0, 0, 0, 0,| phi_0(x0), phi_1(x0), phi_2(x0), phi_3(x0),| 0, 0, 0, 0) 
    then we can get the global J'(U), and this is the rhs of the global dual system. 
   */
      rhs_dual.reinit (dof_handler.n_dofs());
      rhs_dual = 0;

      const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell; 
      Point<2> ref_point;             //used to store the corresponding point on reference cell  
      std::vector<types::global_dof_index> dof_indices (dofs_per_cell); //used to store the global_dof_index

      typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for(; cell!=endc; ++cell)
         if(cell->point_inside(evaluation_point)){     
            for(unsigned int i=0; i<dofs_per_cell; ++i){
         
         /*for current dof_index, i.e., for current base function phi(i)(it's a vector function),
         we may ask, which component of the current base function is nonzero? so we can know what quatity 
         this base function represents.
         for example, for the 0-th base function(whose coefficient is local_dof_values[0]),
         we can get c == 0, which means that this function's 0-th component is non-zero, and is responsible
         of approximating the 0-th component of fe, i.e., the component rho*u
         */
               cell->get_dof_indices (dof_indices);    //get global_indices of the corresponding local_indices
               const unsigned int c = dof_handler.get_fe().system_to_component_index(i).first;
               ref_point = mapping.transform_real_to_unit_cell(cell, evaluation_point);
               if(c==2){      //so that this component is rho
                  rhs_dual(dof_indices[i]) = dof_handler.get_fe().shape_value(i, ref_point);
               }
            }
         }   
     // AssertThrow (false, ExcEvaluationPointNotFound(evaluation_point));   
   }

//==================================================================================
//objective functional is the drag over naca airfoil face.  
//==================================================================================
   //we use parameters here in order to get the corresponding boundary_type of a specific boundary_id,
   //which is specified by user and known to parameters.
   template<int dim>
   DragEvaluation<dim>::
   DragEvaluation(const Parameters::AllParameters<dim>& parameters)
                 :
                 parameters(parameters),
                 naca_boundary_id(-1),
                 farfield_id(-1)
   {
        //firstly we figure out which boundary_id denote the wall_boundary of naca airfoil
         for(unsigned int boundary_id = 0; boundary_id < parameters.max_n_boundaries; ++boundary_id){
            typename EulerEquations<dim>::BoundaryKind boundary_kind = 
                                        parameters.boundary_conditions[boundary_id].kind; 
            if(boundary_kind == EulerEquations<dim>::BoundaryKind::no_penetration_boundary) 
               naca_boundary_id = boundary_id;
            if(boundary_kind == EulerEquations<dim>::BoundaryKind::farfield_boundary)
               farfield_id = boundary_id; 
         }
         //check the naca_boundary_id is correctly assigned
         std::cout<<"naca_boundary_id = "<< naca_boundary_id << std::endl;
         std::cout<<"farfield_id = "<< farfield_id << std::endl;
         //if (naca_boundary_id<0 || ) std::cout<<"the id of wall_boundary is not correct! ";
   }

   //note: in the following dof_handler and primal_solution are both in dual_space
   template<int dim>
   void 
   DragEvaluation<dim>::assemble_rhs(const dealii::DoFHandler<dim>& dof_handler,
                                     const dealii::Mapping<dim>&    mapping,
                                     const dealii::Vector<double>&  primal_solution,
                                     dealii::Vector<double>&        rhs_dual) const 
   {
      AssertThrow(dim==2, ExcNotImplemented());

      rhs_dual.reinit (dof_handler.n_dofs());

      const double alpha = 1.25*Pi/180;   //CHECK if the inflow angle is 2 degree !
      //psi indicate the inflow direction vector
      Tensor<1,dim> psi;               //rank 1 tensor in 2d space, i.e. a vector with 2 components
      psi[0]=cos(alpha);
      psi[1]=sin(alpha);  

      double kinetic_energy_far = 0;
 
      double J_total=0;
      const QGauss<dim-1>           face_quadrature(dof_handler.get_fe().degree+1);  //this is the only thing can be determined by user.

      { //compute free stream kinetic energy
      dealii::Vector<double> W_far(EulerEquations<dim>::n_components);

      parameters.boundary_conditions[farfield_id].values //function parser
                .vector_value(dealii::Point<dim>(0,0), W_far);

      kinetic_energy_far = EulerEquations<dim>::template  
                           compute_kinetic_energy<double>(W_far);
      }

      FEFaceValues<dim> fe_boundary_face (mapping,
                                          dof_handler.get_fe(),     //here fe is a FESystem
                                          face_quadrature,                             
                                          update_values|
                                          update_normal_vectors|
                                          update_quadrature_points|
                                          update_JxW_values);
      typename DoFHandler<dim>::active_cell_iterator cell=dof_handler.begin_active(),
                                                     endc=dof_handler.end();
      //loop over the boundary of the airfoil to compute drag
      for(;cell!=endc;++cell){ 
         for(unsigned int face_no=0;face_no<GeometryInfo<dim>::faces_per_cell;++face_no)
            if(cell->face(face_no)->at_boundary()==true)
            {
               const unsigned int& boundary_id = cell->face(face_no)->boundary_id();
               
               //compute drag
               if (boundary_id == naca_boundary_id)
               { 
                  fe_boundary_face.reinit(cell, face_no);

                  const unsigned int n_q_points = fe_boundary_face.n_quadrature_points;
                  const unsigned int dofs_per_cell = fe_boundary_face.dofs_per_cell;

                  const std::vector<Tensor<1,dim> > &normals
                                          =fe_boundary_face.get_all_normal_vectors();
                  const std::vector<double> &JxW = fe_boundary_face.get_JxW_values();

                  //use dof_indices to store the dof indices of current cell
                  std::vector<types::global_dof_index> dof_indices (dofs_per_cell); 
                  cell->get_dof_indices(dof_indices);

                  //define the independent variables(i.e. the solution variables) and the dependent variable(i.e. the drag)
                  Sacado::Fad::DFad<double> J_cell = 0;   //drag
                  std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);
                  for(unsigned int i=0;i<dofs_per_cell;++i){
                     independent_local_dof_values[i] = primal_solution(dof_indices[i]);
                     independent_local_dof_values[i].diff(i,dofs_per_cell);
                  }

                  //get the conservative variables on the quadrature_points and compute pressures
                  Table<2,Sacado::Fad::DFad<double>> W(n_q_points,
                                                       EulerEquations<dim>::n_components);
                  Table<2,Sacado::Fad::DFad<double>> W_wall(n_q_points,
                                                           EulerEquations<dim>::n_components);

                  std::vector<Sacado::Fad::DFad<double>> pressure(n_q_points);  
             
                  for(unsigned int point=0;point<n_q_points;++point)
                  {
                     for(unsigned int i=0; i<dofs_per_cell; ++i)
                     {
                        const unsigned int c = fe_boundary_face.get_fe()
                                               .system_to_component_index(i).first;
                        W[point][c] += independent_local_dof_values[i]*
                                       fe_boundary_face.shape_value_component(i,point,c);
                     }
                     //in order to derive adjoint consistent discretization, here we must employ a modified
                     //target functional J(uh) = J(u-(uh)). where u-(u) = u is consistent.
                     EulerEquations<dim>::compute_W_wall(normals[point],
                                                         W[point],
                                                         W_wall[point]);

                     pressure[point] = EulerEquations<dim>::template  
                                compute_pressure<Sacado::Fad::DFad<double>>(W_wall[point]);
                   //note: here must add 'template', otherwise will cause error.
                  }  

                  //get the drag contribution from current cell_boundary_face
                  for(unsigned int point=0;point<n_q_points;++point)                 
                     J_cell += normals[point] * 
                               (psi / kinetic_energy_far) *
                               pressure[point] *
                               JxW[point];

                  for(unsigned int k=0;k<dofs_per_cell;++k)
                     rhs_dual(dof_indices[k]) = J_cell.fastAccessDx(k);  

                  J_total += J_cell.val(); 

               }//"if at no_penetration_boundary" end
            }//"if at boundary" end
      }//cell loop end 

      std::cout<<"the total drag is: "<<J_total<<std::endl;
   }

//template class DualFunctionalBase<2>;
template class PointValueEvaluation<2>;
template class DragEvaluation<2>;
} /*namespace DualFunctional*/


